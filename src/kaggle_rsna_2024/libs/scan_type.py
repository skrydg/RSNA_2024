from enum import Enum

class ScanType(Enum):
    sagittal_t2_stir = 0
    sagittal_t1 = 1
    axial_t2 = 2


def string_to_scan_type(str):
    if str == "Sagittal T2/STIR":
        return ScanType.sagittal_t2_stir

    if str == "Sagittal T1":
        return ScanType.sagittal_t1

    if str == "Axial T2":
        return ScanType.sagittal_t1

    assert(0)