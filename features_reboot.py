#!/usr/bin/python

import re
import lief
import hashlib
import numpy as np
import pefile
import distorm3
from sklearn.feature_extraction import FeatureHasher
from ast import literal_eval

LIEF_MAJOR, LIEF_MINOR, _ = lief.__version__.split('.')
LIEF_EXPORT_OBJECT = int(LIEF_MAJOR) > 0 or (int(LIEF_MAJOR) == 0 and int(LIEF_MINOR) >= 10)


class has_debug:
    i_has_debug = 0


class has_rich_header:
    i_has_rich_header = 0


class has_resources:
    i_has_resources = 0


class FeatureType(object):
    ''' Base class from which each feature type may inherit '''

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, bytez, lief_binary):
        ''' Generate a JSON-able representation of the file '''
        raise (NotImplementedError)

    def process_raw_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        raise (NotImplementedError)

    def feature_vector(self, bytez, lief_binary):
        ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. '''
        return self.process_raw_features(self.raw_features(bytez, lief_binary))


class ByteHistogram(FeatureType):
    ''' Byte histogram (count + non-normalized) over the entire binary file '''

    name = 'histogram'
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


class ByteEntropyHistogram(FeatureType):
    ''' 2d byte/entropy histogram based loosely on (Saxe and Berlin, 2015).
    This roughly approximates the joint probability of byte value and local entropy.
    See Section 2.1.1 in https://arxiv.org/pdf/1508.03096.pdf for more info.
    '''

    name = 'byteentropy'
    dim = 256

    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # coarse histogram, 16 bytes per bin
        c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(
            p[wh])) * 2  # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits)

        Hbin = int(H * 2)  # up to 16 bins (max entropy is 8 bits)
        if Hbin == 16:  # handle entropy = 8.0 bits
            Hbin = 15

        return Hbin, c

    def raw_features(self, bytez, lief_binary):
        output = np.zeros((16, 16), dtype=np.int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            # strided trick from here: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]

            # from the blocks, compute histogram
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c

        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


class SectionInfo(FeatureType):
    ''' Information about section names, sizes and entropy.  Uses hashing trick
    to summarize all this section info into a feature vector.
    '''

    name = 'section'
    dim = 5 + 50 + 50 + 50 + 50 + 50

    def __init__(self):
        super(FeatureType, self).__init__()

    @staticmethod
    def _properties(s):
        return [str(c).split('.')[-1] for c in s.characteristics_lists]

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {"entry": "", "sections": []}

        # properties of entry point, or if invalid, the first executable section
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except lief.not_found:
            # bad entry point, let's find the first executable section
            entry_section = ""
            for s in lief_binary.sections:
                if lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in s.characteristics_lists:
                    entry_section = s.name
                    break

        raw_obj = {"entry": entry_section}
        raw_obj["sections"] = [{
            'name': s.name,
            'size': s.size,
            'entropy': s.entropy,
            'vsize': s.virtual_size,
            'props': self._properties(s)
        } for s in lief_binary.sections]
        return raw_obj

    def process_raw_features(self, raw_obj):
        sections = raw_obj['sections']
        general = [
            len(sections),  # total number of sections
            # number of sections with nonzero size
            sum(1 for s in sections if s['size'] == 0),
            # number of sections with an empty name
            sum(1 for s in sections if s['name'] == ""),
            # number of RX
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            # number of W
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        # gross characteristics of each section
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([raw_obj['entry']]).toarray()[0]
        characteristics = [p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
            characteristics_hashed
        ]).astype(np.float32)


class ImportsInfo(FeatureType):
    ''' Information about imported libraries and functions from the
    import address table.  Note that the total number of imported
    functions is contained in GeneralFileInfo.
    '''

    name = 'imports'
    dim = 1280

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        imports = {}
        if lief_binary is None:
            return imports

        for lib in lief_binary.imports:
            if lib.name not in imports:
                imports[lib.name] = []  # libraries can be duplicated in listing, extend instead of overwrite

            # Clipping assumes there are diminishing returns on the discriminatory power of imported functions
            #  beyond the first 10000 characters, and this will help limit the dataset size
            for entry in lib.entries:
                if entry.is_ordinal:
                    imports[lib.name].append("ordinal" + str(entry.ordinal))
                else:
                    imports[lib.name].append(entry.name[:10000])

        return imports

    def process_raw_features(self, raw_obj):
        # unique libraries
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]

        # A string like "kernel32.dll:CreateFileMappingA" for each imported function
        imports = [lib.lower() + ':' + e for lib, elist in raw_obj.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]

        # Two separate elements: libraries (alone) and fully-qualified names of imported functions
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)


class ExportsInfo(FeatureType):
    ''' Information about exported functions. Note that the total number of exported
    functions is contained in GeneralFileInfo.
    '''

    name = 'exports'
    dim = 128

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return []

        # Clipping assumes there are diminishing returns on the discriminatory power of exports beyond
        #  the first 10000 characters, and this will help limit the dataset size
        if LIEF_EXPORT_OBJECT:
            # export is an object with .name attribute (0.10.0 and later)
            clipped_exports = [export.name[:10000] for export in lief_binary.exported_functions]
        else:
            # export is a string (LIEF 0.9.0 and earlier)
            clipped_exports = [export[:10000] for export in lief_binary.exported_functions]

        return clipped_exports

    def process_raw_features(self, raw_obj):
        exports_hashed = FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0]
        return exports_hashed.astype(np.float32)


class GeneralFileInfo(FeatureType):
    ''' General information about the file '''

    name = 'general'
    dim = 10

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            has_debug.i_has_debug = 0
            has_resources.i_has_resources = 0
            has_rich_header.i_has_rich_header = 0
            return {
                'size': len(bytez),
                'vsize': 0,
                'has_debug': 0,
                'exports': 0,
                'imports': 0,
                'has_relocations': 0,
                'has_resources': 0,
                'has_signature': 0,
                'has_tls': 0,
                'symbols': 0
            }

        has_debug.i_has_debug = int(lief_binary.has_debug)
        has_rich_header.i_has_rich_header = int(lief_binary.has_rich_header)
        has_resources.i_has_resources = int(lief_binary.has_resources)

        return {
            'size': len(bytez),
            'vsize': lief_binary.virtual_size,
            'has_debug': int(lief_binary.has_debug),
            'exports': len(lief_binary.exported_functions),
            'imports': len(lief_binary.imported_functions),
            'has_relocations': int(lief_binary.has_relocations),
            'has_resources': int(lief_binary.has_resources),
            'has_signature': int(lief_binary.has_signature),
            'has_tls': int(lief_binary.has_tls),
            'symbols': len(lief_binary.symbols),
        }

    def process_raw_features(self, raw_obj):
        return np.asarray([
            raw_obj['size'], raw_obj['vsize'], raw_obj['has_debug'], raw_obj['exports'], raw_obj['imports'],
            raw_obj['has_relocations'], raw_obj['has_resources'], raw_obj['has_signature'], raw_obj['has_tls'],
            raw_obj['symbols']
        ],
            dtype=np.float32)


class HeaderFileInfo(FeatureType):
    ''' Machine, architecure, OS, linker and other information extracted from header '''

    name = 'header'
    dim = 92

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj['coff'] = {'timestamp': 0, 'machine': "", 'characteristics': []}
        raw_obj['optional'] = {
            'subsystem': "",
            'dll_characteristics': [],
            'magic': "",
            'major_image_version': 0,
            'minor_image_version': 0,
            'major_linker_version': 0,
            'minor_linker_version': 0,
            'major_operating_system_version': 0,
            'minor_operating_system_version': 0,
            'major_subsystem_version': 0,
            'minor_subsystem_version': 0,
            'sizeof_code': 0,
            'sizeof_headers': 0,
            'sizeof_heap_commit': 0
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['coff']['timestamp'] = lief_binary.header.time_date_stamps
        raw_obj['coff']['machine'] = str(lief_binary.header.machine).split('.')[-1]
        raw_obj['coff']['characteristics'] = [str(c).split('.')[-1] for c in lief_binary.header.characteristics_list]
        raw_obj['optional']['subsystem'] = str(lief_binary.optional_header.subsystem).split('.')[-1]
        raw_obj['optional']['dll_characteristics'] = [
            str(c).split('.')[-1] for c in lief_binary.optional_header.dll_characteristics_lists
        ]
        raw_obj['optional']['magic'] = str(lief_binary.optional_header.magic).split('.')[-1]
        raw_obj['optional']['major_image_version'] = lief_binary.optional_header.major_image_version
        raw_obj['optional']['minor_image_version'] = lief_binary.optional_header.minor_image_version
        raw_obj['optional']['major_linker_version'] = lief_binary.optional_header.major_linker_version
        raw_obj['optional']['minor_linker_version'] = lief_binary.optional_header.minor_linker_version
        raw_obj['optional'][
            'major_operating_system_version'] = lief_binary.optional_header.major_operating_system_version
        raw_obj['optional'][
            'minor_operating_system_version'] = lief_binary.optional_header.minor_operating_system_version
        raw_obj['optional']['major_subsystem_version'] = lief_binary.optional_header.major_subsystem_version
        raw_obj['optional']['minor_subsystem_version'] = lief_binary.optional_header.minor_subsystem_version
        raw_obj['optional']['sizeof_code'] = lief_binary.optional_header.sizeof_code
        raw_obj['optional']['sizeof_headers'] = lief_binary.optional_header.sizeof_headers
        raw_obj['optional']['sizeof_heap_commit'] = lief_binary.optional_header.sizeof_heap_commit
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['coff']['timestamp'],
            FeatureHasher(16, input_type="string").transform([[raw_obj['coff']['machine']]]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['coff']['characteristics']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([[raw_obj['optional']['subsystem']]]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['optional']['dll_characteristics']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([[raw_obj['optional']['magic']]]).toarray()[0],
            raw_obj['optional']['major_image_version'],
            raw_obj['optional']['minor_image_version'],
            raw_obj['optional']['major_linker_version'],
            raw_obj['optional']['minor_linker_version'],
            raw_obj['optional']['major_operating_system_version'],
            raw_obj['optional']['minor_operating_system_version'],
            raw_obj['optional']['major_subsystem_version'],
            raw_obj['optional']['minor_subsystem_version'],
            raw_obj['optional']['sizeof_code'],
            raw_obj['optional']['sizeof_headers'],
            raw_obj['optional']['sizeof_heap_commit'],
        ]).astype(np.float32)


class StringExtractor(FeatureType):
    ''' Extracts strings from raw byte stream '''

    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

    def __init__(self):
        super(FeatureType, self).__init__()
        # all consecutive runs of 0x20 - 0x7f that are 5+ characters
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        # occurances of the string 'C:\'.  Not actually extracting the path
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # occurances of http:// or https://.  Not actually extracting the URLs
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # occurances of the string prefix HKEY_.  No actually extracting registry names
        self._registry = re.compile(b'HKEY_')
        # crude evidence of an MZ header (dropper?) somewhere in the byte stream
        self._mz = re.compile(b'MZ')

    def raw_features(self, bytez, lief_binary):
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0

        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),  # store non-normalized histogram
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            np.asarray(raw_obj['printabledist']) / hist_divisor, raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)


class DataDirectories(FeatureType):
    ''' Extracts size and virtual address of the first 15 data directories '''

    name = 'datadirectories'
    dim = 15 * 2

    def __init__(self):
        super(FeatureType, self).__init__()
        self._name_order = [
            "EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
            "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE",
            "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"
        ]

    def raw_features(self, bytez, lief_binary):
        output = []
        if lief_binary is None:
            return output

        for data_directory in lief_binary.data_directories:
            output.append({
                "name": str(data_directory.type).replace("DATA_DIRECTORY.", ""),
                "size": data_directory.size,
                "virtual_address": data_directory.rva
            })
        return output

    def process_raw_features(self, raw_obj):
        features = np.zeros(2 * len(self._name_order), dtype=np.float32)
        for i in range(len(self._name_order)):
            if i < len(raw_obj):
                features[2 * i] = raw_obj[i]["size"]
                features[2 * i + 1] = raw_obj[i]["virtual_address"]
        return features


class TLSInfo(FeatureType):
    name = 'TLS'
    dim = 99

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        ## add the feature
        raw_obj = {
            'addressof_callbacks': 0,
            'addressof_index': 0,
            'addressof_raw_data': [0, 0],
            'callbacks': [],
            'characteristics': 0,
            'data_template': [],
            'has_data_directory': 0,
            'directory_has_section': 0,
            'directory_rva': 0,
            'directory_size': 0,
            'directory_type': "",
            'has_section': 0,
            'section_name': "",
            'section_size': 0,
            'section_entropy': 0,
            'section_vsize': 0,
            'section_pointer_to_raw_data': 0,
            'section_pointer_to_relocations': 0,
            'section_pointer_to_line_numbers': 0,
            'section_number_of_relocations': 0,
            'section_number_of_line_numbers': 0,
            'section_characteristics': [],
            'sizeof_zero_fill': 0,
        }

        if lief_binary is None:
            return raw_obj

        raw_obj['addressof_callbacks'] = lief_binary.tls.addressof_callbacks
        raw_obj['addressof_index'] = lief_binary.tls.addressof_index
        raw_obj['addressof_raw_data'] = list(lief_binary.tls.addressof_raw_data)  # tuple to list
        raw_obj['callbacks'] = [str(c).split('.')[-1] for c in lief_binary.tls.callbacks]
        raw_obj['characteristics'] = lief_binary.tls.characteristics
        raw_obj['data_template'] = [str(c).split('.')[-1] for c in lief_binary.tls.data_template]
        raw_obj['has_data_directory'] = lief_binary.tls.has_data_directory
        raw_obj['has_section'] = lief_binary.tls.has_section
        raw_obj['sizeof_zero_fill'] = lief_binary.tls.sizeof_zero_fill

        if lief_binary.tls.has_data_directory is True:
            raw_obj['directory_has_section'] = lief_binary.tls.directory.has_section
            raw_obj['directory_rva'] = lief_binary.tls.directory.rva
            raw_obj['directory_size'] = lief_binary.tls.directory.size
            raw_obj['directory_type'] = str(lief_binary.tls.directory.type).split('.')[-1]

        if lief_binary.tls.has_section is True:
            raw_obj['section_name'] = lief_binary.tls.section.name
            raw_obj['section_size'] = lief_binary.tls.section.size
            raw_obj['section_entropy'] = lief_binary.tls.section.entropy
            raw_obj['section_vsize'] = lief_binary.tls.section.virtual_size
            raw_obj['section_pointer_to_raw_data'] = lief_binary.tls.section.pointerto_raw_data
            raw_obj['section_pointer_to_relocations'] = lief_binary.tls.section.pointerto_relocation
            raw_obj['section_pointer_to_line_numbers'] = lief_binary.tls.section.pointerto_line_numbers
            raw_obj['section_number_of_relocations'] = lief_binary.tls.section.numberof_relocations
            raw_obj['section_number_of_line_numbers'] = lief_binary.tls.section.numberof_line_numbers
            raw_obj['section_characteristics'] = [str(i).split('.')[-1] for i in
                                                  lief_binary.tls.section.characteristics_lists]

        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['addressof_callbacks'],
            raw_obj['addressof_index'],
            raw_obj['addressof_raw_data'],  # dim 2
            FeatureHasher(16, input_type="string").transform([raw_obj['callbacks']]).toarray()[0],
            raw_obj['characteristics'],
            FeatureHasher(16, input_type="string").transform([raw_obj['data_template']]).toarray()[0],
            raw_obj['has_data_directory'],
            raw_obj['directory_has_section'],
            raw_obj['directory_rva'],
            raw_obj['directory_size'],
            FeatureHasher(16, input_type="string").transform([raw_obj['directory_type']]).toarray()[0],
            raw_obj['has_section'],
            FeatureHasher(16, input_type="string").transform([raw_obj['section_name']]).toarray()[0],
            raw_obj['section_size'],
            raw_obj['section_entropy'],
            raw_obj['section_vsize'],
            raw_obj['section_pointer_to_raw_data'],
            raw_obj['section_pointer_to_relocations'],
            raw_obj['section_pointer_to_line_numbers'],
            raw_obj['section_number_of_relocations'],
            raw_obj['section_number_of_line_numbers'],
            FeatureHasher(16, input_type="string").transform([raw_obj['section_characteristics']]).toarray()[0],
            raw_obj['sizeof_zero_fill'],
        ]).astype(np.float32)


class SignatureInfo(FeatureType):
    name = 'signature'
    dim = 18

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj = {
            'digest_algorithm': "",
            'version': 0,
            'numberof_certificates': 0,
        }
        if lief_binary is None:
            return raw_obj
        raw_obj['digest_algorithm'] = lief_binary.signature.digest_algorithm
        raw_obj['version'] = lief_binary.signature.version
        raw_obj['numberof_certificates'] = len(lief_binary.signature.certificates)
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            FeatureHasher(16, input_type="string").transform([raw_obj['digest_algorithm']]).toarray()[0],
            raw_obj['version'],
            raw_obj['numberof_certificates']
        ]).astype(np.float32)


class ContentInfo(FeatureType):
    name = 'content_info'
    dim = 64

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj = {
            'content_type': "",
            'digest': [],
            'digest_algorithm': "",
            'type': ""
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['content_type'] = lief_binary.signature.content_info.content_type
        raw_obj['digest'] = lief_binary.signature.content_info.digest
        raw_obj['digest_algorithm'] = lief_binary.signature.content_info.digest_algorithm
        raw_obj['type'] = lief_binary.signature.content_info.type
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            FeatureHasher(16, input_type="string").transform([raw_obj['content_type']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['digest']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['digest_algorithm']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['type']]).toarray()[0]
        ]).astype(np.float32)


class SignerInfo(FeatureType):
    name = 'signer_info'
    dim = 33  # + 16 + 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj = {
            'digest_algorithm': "",
            # 'encrypted_digest': [],
            # 'issuer': ('', []),
            'signature_algorithm': "",
            'version': 0
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['digest_algorithm'] = lief_binary.signature.signer_info.digest_algorithm
        raw_obj['encrypted_digest'] = lief_binary.signature.signer_info.encrypted_digest
        # raw_obj['issuer'] = lief_binary.signature.signer_info.issuer
        # raw_obj['signature_algorithm'] = lief_binary.signature.signer_info.signature_algorithm
        raw_obj['version'] = lief_binary.signature.signer_info.version

        if raw_obj['encrypted_digest'] == []:
            raw_obj['encrypted_digest'] = [0 for i in range(256)]

        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            FeatureHasher(16, input_type="string").transform([raw_obj['digest_algorithm']]).toarray()[0],
            # raw_obj['encrypted_digest'], # dim 256
            # FeatureHasher(16, input_type="string").transform([raw_obj['issuer']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['signature_algorithm']]).toarray()[0],
            raw_obj['version']
        ]).astype(np.float32)


class DosHeaderInfo(FeatureType):
    name = 'dos_header'
    dim = 16 + 32

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj['dos_header'] = {
            'addressof_new_exeheader': 0,
            'addressof_relocation_table': 0,
            'checksum': 0,
            'file_size_in_pages': 0,
            'header_size_in_paragraphs': 0,
            'initial_ip': 0,
            'initial_relative_cs': 0,
            'initial_relative_ss': 0,
            'initial_sp': 0,
            'magic': "",
            'maximum_extra_paragraphs': 0,
            'minimum_extra_paragraphs': 0,
            'numberof_relocation': 0,
            'oem_id': 0,
            'oem_info': 0,
            'overlay_number': 0,
            'used_bytes_in_the_last_page': 0,
        }
        raw_obj['dos_stub'] = {
            'dos_stub': [],
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['dos_header']['addressof_new_exeheader'] = lief_binary.dos_header.addressof_new_exeheader
        raw_obj['dos_header']['addressof_relocation_table'] = lief_binary.dos_header.addressof_relocation_table
        raw_obj['dos_header']['checksum'] = lief_binary.dos_header.checksum
        raw_obj['dos_header']['file_size_in_pages'] = lief_binary.dos_header.file_size_in_pages
        raw_obj['dos_header']['header_size_in_paragraphs'] = lief_binary.dos_header.header_size_in_paragraphs
        raw_obj['dos_header']['initial_ip'] = lief_binary.dos_header.initial_ip
        raw_obj['dos_header']['initial_relative_cs'] = lief_binary.dos_header.initial_relative_cs
        raw_obj['dos_header']['initial_relative_ss'] = lief_binary.dos_header.initial_relative_ss
        raw_obj['dos_header']['initial_sp'] = lief_binary.dos_header.initial_sp
        raw_obj['dos_header']['magic'] = str(lief_binary.dos_header.magic).split('.')[-1]
        raw_obj['dos_header']['maximum_extra_paragraphs'] = lief_binary.dos_header.maximum_extra_paragraphs
        raw_obj['dos_header']['minimum_extra_paragraphs'] = lief_binary.dos_header.minimum_extra_paragraphs
        raw_obj['dos_header']['numberof_relocation'] = lief_binary.dos_header.numberof_relocation
        raw_obj['dos_header']['oem_id'] = lief_binary.dos_header.oem_id
        raw_obj['dos_header']['oem_info'] = lief_binary.dos_header.oem_info
        raw_obj['dos_header']['overlay_number'] = lief_binary.dos_header.overlay_number
        raw_obj['dos_header']['used_bytes_in_the_last_page'] = lief_binary.dos_header.used_bytes_in_the_last_page
        raw_obj['dos_stub']['dos_stub'] = [str(c).split('.')[-1] for c in lief_binary.dos_stub]
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['dos_header']['addressof_new_exeheader'],
            raw_obj['dos_header']['addressof_relocation_table'],
            raw_obj['dos_header']['checksum'],
            raw_obj['dos_header']['file_size_in_pages'],
            raw_obj['dos_header']['header_size_in_paragraphs'],
            raw_obj['dos_header']['initial_ip'],
            raw_obj['dos_header']['initial_relative_cs'],
            raw_obj['dos_header']['initial_relative_ss'],
            raw_obj['dos_header']['initial_sp'],
            FeatureHasher(16, input_type="string").transform([raw_obj['dos_header']['magic']]).toarray()[0],
            raw_obj['dos_header']['maximum_extra_paragraphs'],
            raw_obj['dos_header']['minimum_extra_paragraphs'],
            raw_obj['dos_header']['numberof_relocation'],
            raw_obj['dos_header']['oem_id'],
            raw_obj['dos_header']['oem_info'],
            raw_obj['dos_header']['overlay_number'],
            raw_obj['dos_header']['used_bytes_in_the_last_page'],
            FeatureHasher(16, input_type="string").transform([raw_obj['dos_stub']['dos_stub']]).toarray()[0],
        ]).astype(np.float32)


class DebugInfo(FeatureType):
    name = 'debug'
    dim = 23

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj['debug'] = {
            'characteristics': 0,
            'has_code_view': 0,
            'major_version': 0,
            'minor_version': 0,
            'pointerto_rawdata': 0,
            'sizeof_data': 0,
            'timestamp': 0,
            'type': "",
        }
        if lief_binary is None or lief_binary.has_debug == 0:
            return raw_obj
        try:
            raw_obj['debug']['characteristics'] = lief_binary.debug.characteristics
            raw_obj['debug']['has_code_view'] = lief_binary.debug.has_code_view
            raw_obj['debug']['major_version'] = lief_binary.debug.major_version
            raw_obj['debug']['minor_version'] = lief_binary.debug.minor_version
            raw_obj['debug']['pointerto_rawdata'] = lief_binary.debug.pointerto_rawdata
            raw_obj['debug']['sizeof_data'] = lief_binary.debug.sizeof_data
            raw_obj['debug']['timestamp'] = lief_binary.debug.timestamp
            raw_obj['debug']['type'] = str(lief_binary.debug.type).split('.')[-1]
        except:
            pass
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['debug']['characteristics'],
            raw_obj['debug']['has_code_view'],
            raw_obj['debug']['major_version'],
            raw_obj['debug']['minor_version'],
            raw_obj['debug']['pointerto_rawdata'],
            raw_obj['debug']['sizeof_data'],
            raw_obj['debug']['timestamp'],
            FeatureHasher(16, input_type="string").transform([[raw_obj['debug']['type']]]).toarray()[0],
        ]).astype(np.float32)


class RichHeaderInfo(FeatureType):
    name = 'rich_header'
    dim = 4  # dim is feature number

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        ## add the feature
        raw_obj['rich_header'] = {
            'id': [],
            'build_id': [],
            'count': [],
            'key': 0,
        }

        if lief_binary is None or lief_binary.has_rich_header == 0:
            return raw_obj

        for i in range(len(lief_binary.rich_header.entries)):
            raw_obj['rich_header']['id'].append(lief_binary.rich_header.entries[i].id)
            raw_obj['rich_header']['build_id'].append(lief_binary.rich_header.entries[i].build_id)
            raw_obj['rich_header']['count'].append(lief_binary.rich_header.entries[i].count)
            raw_obj['rich_header']['key'] = lief_binary.rich_header.key
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['rich_header']['id'],
            raw_obj['rich_header']['build_id'],
            raw_obj['rich_header']['count'],
            raw_obj['rich_header']['key'],
        ]).astype(np.float32)


class ResourceInfo(FeatureType):
    name = 'resources'
    dim = 26

    def __init__(self):
        super(FeatureType, self).__init__()

    # def resource_child(childs):
    #     if childs.is_directory == 1:
    #         raw_obj = {
    #             'childs': [],
    #             'characteristics': 0,
    #             'depth': 0,
    #             'has_name': False,
    #             'id': 0,
    #             'is_data': False,
    #             'is_directory': False,
    #             'major_version': 0,
    #             'minor_version': 0,
    #             'name': "",
    #             'numberof_id_entries': 0,
    #             'numberof_name_entries': 0,
    #             'time_date_stamp': 0,
    #         }
    #         for i in range(len(childs.childs)):
    #             raw_obj['childs'].append(ResourceInfo.resource_child(childs.childs[i]))
    #         raw_obj['characteristics'] = childs.characteristics
    #         raw_obj['depth'] = childs.depth
    #         raw_obj['has_name'] = childs.has_name
    #         raw_obj['id'] = childs.id
    #         raw_obj['is_data'] = childs.is_data
    #         raw_obj['is_directory'] = childs.is_directory
    #         raw_obj['major_version'] = childs.major_version
    #         raw_obj['minor_version'] = childs.minor_version
    #         raw_obj['name'] = childs.name
    #         raw_obj['numberof_id_entries'] = childs.numberof_id_entries
    #         raw_obj['numberof_name_entries'] = childs.numberof_name_entries
    #         raw_obj['time_date_stamp'] = childs.time_date_stamp
    #         return raw_obj
    #
    #     elif childs.is_data == 1:
    #         raw_obj = {
    #             'code_page': 0,
    #             'depth': 0,
    #             'has_name': False,
    #             'id': 0,
    #             'is_data': False,
    #             'is_directory': False,
    #             'name': "",
    #             'offset': 0,
    #         }
    #         raw_obj['code_page'] = childs.code_page
    #         raw_obj['depth'] = childs.depth
    #         raw_obj['has_name'] = childs.has_name
    #         raw_obj['id'] = childs.id
    #         raw_obj['is_data'] = childs.is_data
    #         raw_obj['is_directory'] = childs.is_directory
    #         raw_obj['name'] = childs.name
    #         raw_obj['offset'] = childs.offset
    #         return raw_obj
    #     else:
    #         print("unidentified error! - resource_child()")
    #         exit()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj['resources'] = {
            # 'childs': [],
            'characteristisc': 0,
            'depth': 0,
            'has_name': False,
            'id': 0,
            'is_data': False,
            'is_directory': False,
            'major_version': 0,
            'minor_version': 0,
            'name': "",
            'numberof_id_entries': 0,
            'numberof_name_entries': 0,
            'time_date_stamp': 0,
        }
        if lief_binary is None or lief_binary.has_resources == 0:
            return raw_obj

        # for i in range(len(lief_binary.resources.childs)):
            # raw_obj['resources']['childs'].append(ResourceInfo.resource_child(lief_binary.resources.childs[i]))
        raw_obj['resources']['characteristics'] = lief_binary.resources.characteristics
        raw_obj['resources']['depth'] = lief_binary.resources.depth
        raw_obj['resources']['has_name'] = lief_binary.resources.has_name
        raw_obj['resources']['id'] = lief_binary.resources.id
        raw_obj['resources']['is_data'] = lief_binary.resources.is_data
        raw_obj['resources']['is_directory'] = lief_binary.resources.is_directory
        raw_obj['resources']['major_version'] = lief_binary.resources.major_version
        raw_obj['resources']['minor_version'] = lief_binary.resources.minor_version
        raw_obj['resources']['name'] = lief_binary.resources.name
        raw_obj['resources']['numberof_id_entries'] = lief_binary.resources.numberof_id_entries
        raw_obj['resources']['numberof_name_entries'] = lief_binary.resources.numberof_name_entries
        raw_obj['resources']['time_date_stamp'] = lief_binary.resources.time_date_stamp
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            # FeatureHasher(16, input_type="dict").transform([raw_obj['resources']['childs']]).toarray()[0],
            raw_obj['resources']['depth'],
            raw_obj['resources']['has_name'],
            raw_obj['resources']['id'],
            raw_obj['resources']['is_data'],
            raw_obj['resources']['is_directory'],
            raw_obj['resources']['major_version'],
            raw_obj['resources']['minor_version'],
            FeatureHasher(16, input_type="string").transform([raw_obj['resources']['name']]).toarray()[0],
            raw_obj['resources']['numberof_id_entries'],
            raw_obj['resources']['numberof_name_entries'],
            raw_obj['resources']['time_date_stamp'],
        ]).astype(np.float32)


class OpcodeInfo(FeatureType):
    name = 'opcode'

    dim = 210

    section_characteristics = [
        ('IMAGE_SCN_TYPE_REG', 0x00000000),  # reserved
        ('IMAGE_SCN_TYPE_DSECT', 0x00000001),  # reserved
        ('IMAGE_SCN_TYPE_NOLOAD', 0x00000002),  # reserved
        ('IMAGE_SCN_TYPE_GROUP', 0x00000004),  # reserved
        ('IMAGE_SCN_TYPE_NO_PAD', 0x00000008),  # reserved
        ('IMAGE_SCN_TYPE_COPY', 0x00000010),  # reserved

        ('IMAGE_SCN_CNT_CODE', 0x00000020),
        ('IMAGE_SCN_CNT_INITIALIZED_DATA', 0x00000040),
        ('IMAGE_SCN_CNT_UNINITIALIZED_DATA', 0x00000080),

        ('IMAGE_SCN_LNK_OTHER', 0x00000100),
        ('IMAGE_SCN_LNK_INFO', 0x00000200),
        ('IMAGE_SCN_LNK_OVER', 0x00000400),  # reserved
        ('IMAGE_SCN_LNK_REMOVE', 0x00000800),
        ('IMAGE_SCN_LNK_COMDAT', 0x00001000),

        ('IMAGE_SCN_MEM_PROTECTED', 0x00004000),  # obsolete
        ('IMAGE_SCN_NO_DEFER_SPEC_EXC', 0x00004000),
        ('IMAGE_SCN_GPREL', 0x00008000),
        ('IMAGE_SCN_MEM_FARDATA', 0x00008000),
        ('IMAGE_SCN_MEM_SYSHEAP', 0x00010000),  # obsolete
        ('IMAGE_SCN_MEM_PURGEABLE', 0x00020000),
        ('IMAGE_SCN_MEM_16BIT', 0x00020000),
        ('IMAGE_SCN_MEM_LOCKED', 0x00040000),
        ('IMAGE_SCN_MEM_PRELOAD', 0x00080000),

        ('IMAGE_SCN_ALIGN_1BYTES', 0x00100000),
        ('IMAGE_SCN_ALIGN_2BYTES', 0x00200000),
        ('IMAGE_SCN_ALIGN_4BYTES', 0x00300000),
        ('IMAGE_SCN_ALIGN_8BYTES', 0x00400000),
        ('IMAGE_SCN_ALIGN_16BYTES', 0x00500000),  # default alignment
        ('IMAGE_SCN_ALIGN_32BYTES', 0x00600000),
        ('IMAGE_SCN_ALIGN_64BYTES', 0x00700000),
        ('IMAGE_SCN_ALIGN_128BYTES', 0x00800000),
        ('IMAGE_SCN_ALIGN_256BYTES', 0x00900000),
        ('IMAGE_SCN_ALIGN_512BYTES', 0x00A00000),
        ('IMAGE_SCN_ALIGN_1024BYTES', 0x00B00000),
        ('IMAGE_SCN_ALIGN_2048BYTES', 0x00C00000),
        ('IMAGE_SCN_ALIGN_4096BYTES', 0x00D00000),
        ('IMAGE_SCN_ALIGN_8192BYTES', 0x00E00000),
        ('IMAGE_SCN_ALIGN_MASK', 0x00F00000),

        ('IMAGE_SCN_LNK_NRELOC_OVFL', 0x01000000),
        ('IMAGE_SCN_MEM_DISCARDABLE', 0x02000000),
        ('IMAGE_SCN_MEM_NOT_CACHED', 0x04000000),
        ('IMAGE_SCN_MEM_NOT_PAGED', 0x08000000),
        ('IMAGE_SCN_MEM_SHARED', 0x10000000),
        ('IMAGE_SCN_MEM_EXECUTE', 0x20000000),
        ('IMAGE_SCN_MEM_READ', 0x40000000),
        ('IMAGE_SCN_MEM_WRITE', 0x80000000)]

    SECTION_CHARACTERISTICS = dict([(e[1], e[0]) for e in section_characteristics] + section_characteristics)

    def __init__(self):
        super(FeatureType, self).__init__()

    def retrieve_flags(self, flag_dict, flag_filter):
        return [(f[0], f[1]) for f in list(flag_dict.items()) if
                isinstance(f[0], (str, bytes)) and f[0].startswith(flag_filter)]

    def get_info(self, sample_path):
        opcode_count = 0
        section_flags = self.retrieve_flags(self.SECTION_CHARACTERISTICS, 'IMAGE_SCN_')
        pe = pefile.PE(sample_path)
        op_list_count = {}

        for section in pe.sections:
            flags = []

            for flag in sorted(section_flags):
                if getattr(section, flag[0]):
                    flags.append(flag[0])
            if 'IMAGE_SCN_MEM_EXECUTE' in flags:
                iterable = distorm3.DecodeGenerator(0, section.get_data(), distorm3.Decode32Bits)

                for (offset, size, instruction, hexdump) in iterable:
                    # print("%.8x: %-32s %s" % (offset, hexdump, instruction.split(" ")[0]))
                    op_code = instruction.split(" ")[0]
                    if op_code not in op_list_count.keys():
                        op_list_count[op_code] = 1
                    elif op_code in op_list_count.keys():
                        op_list_count[op_code] = op_list_count[op_code] + 1

                for flag in sorted(section_flags):
                    if getattr(section, flag[0]):
                        flags.append(flag[0])

        return np.hstack([op_list_count])[0]


class AuthenticatedAttributesInfo(FeatureType):
    name = 'authenticated_attributes'
    dim = 48  # +16(message_digest)

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {
            'content_type': "",
            'more_info': "",
            'program_name': ""
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['content_type'] = lief_binary.signature.signer_info.authenticated_attributes.content_type
        raw_obj['more_info'] = lief_binary.signature.signer_info.authenticated_attributes.more_info
        raw_obj['program_name'] = lief_binary.signature.signer_info.authenticated_attributes.program_name
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            FeatureHasher(16, input_type="string").transform([raw_obj['content_type']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['more_info']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['program_name']]).toarray()[0]
        ]).astype(np.float32)


class PDBInfo(FeatureType):
    name = 'code_view'
    dim = 49

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj['code_view'] = {
            'age': 0,
            'cv_signature': "",
            'filename': "",
            'signature': [],
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['age'] = lief_binary.debug.code_view.age
        raw_obj['cv_signature'] = str(lief_binary.debug.code_view.cv_signature).split('.')[-1]
        raw_obj['filename'] = str(lief_binary.debug.code_view.filename).split('\\')[-1].split('.')[0]
        raw_obj['signature'] = lief_binary.debug.code_view.signature
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['age'],
            FeatureHasher(16, input_type="string").transform([raw_obj['cv_signature']]).toarray()[0],
            FeatureHasher(16, input_type="string").transform([raw_obj['filename']]).toarray()[0],
            raw_obj['signature'],
        ]).astype(np.float32)


class RelocationInfo(FeatureType):
    name = 'relocations'
    dim = 1

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj = {
            'virtual_address': [],
        }

        if lief_binary is None:
            return raw_obj

        for i in range(0, len(lief_binary.relocations)):
            raw_obj['virtual_address'].append(int(lief_binary.relocations[i].virtual_address))
            return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['virtual_address']
        ]).astype(np.float32)


class ByteOneGramInfo(FeatureType):
    name = "byteonegram"

    def __init__(self):
        super(FeatureType, self).__init__()

    def get_info(self, sample_path):

        bytez = open(sample_path, 'rb')

        Byte = bytez.read(1)
        Dec = ord(Byte)
        raw_obj = []
        sum = "{"
        arr = [0 for i in range(256)]

        for j in range(256):
            arr[Dec] += 1
            Dec = ord(bytez.read(1))

        for k in range(256):
            if k < 16:
                sum += "\"0x0" + hex(k)[2:] + "\":" + str(arr[k]) + ', '
            else:
                sum += "\"" + str(hex(k)) + "\":" + str(arr[k]) + ', '

        sum = sum[:-2] + "}"

        bytez.close()

        return (literal_eval(sum))


class PEFeatureExtractor(object):
    ''' Extract useful features from a PE file, and return as a vector of fixed size. '''

    def __init__(self, feature_version=2):
        self.features = [
            ByteHistogram(),
            ByteEntropyHistogram(),
            StringExtractor(),
            GeneralFileInfo(),
            HeaderFileInfo(),
            SectionInfo(),
            ImportsInfo(),
            ExportsInfo(),
            TLSInfo(),
            SignatureInfo(),
            ContentInfo(),
            SignerInfo(),
            DosHeaderInfo(),
            AuthenticatedAttributesInfo(),
            DebugInfo(),
            ResourceInfo(),
            #RichHeaderInfo(),
        ]
        if feature_version == 1:
            if not lief.__version__.startswith("0.8.3"):
                print(f"WARNING: EMBER feature version 1 were computed using lief version 0.8.3-18d5b75")
                print(f"WARNING:   lief version {lief.__version__} found instead. There may be slight inconsistencies")
                print(f"WARNING:   in the feature calculations.")
        elif feature_version == 2:
            self.features.append(DataDirectories())
            if not lief.__version__.startswith("0.9.0"):
                print(f"WARNING: EMBER feature version 2 were computed using lief version 0.9.0-")
                print(f"WARNING:   lief version {lief.__version__} found instead. There may be slight inconsistencies")
                print(f"WARNING:   in the feature calculations.")
        else:
            raise Exception(f"EMBER feature version must be 1 or 2. Not {feature_version}")
        self.dim = sum([fe.dim for fe in self.features])

    def raw_features(self, bytez, sample_path):
        lief_errors = (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error, lief.read_out_of_bound,
                       RuntimeError)
        try:
            lief_binary = lief.PE.parse(list(bytez))
        except lief_errors as e:
            print("lief error: ", str(e))
            lief_binary = None
        except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
            raise

        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features.update({fe.name: fe.raw_features(bytez, lief_binary) for fe in self.features})

        opcode = OpcodeInfo().get_info(sample_path)
        features.update({"opcode": opcode})

        byteonegram = ByteOneGramInfo().get_info(sample_path)
        features.update({"byteonegram": byteonegram})

        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, bytez, sample_path):
        return self.process_raw_features(self.raw_features(bytez, sample_path))