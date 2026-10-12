#!/usr/bin/env python3
"""Repack a downloaded source ZIP without archive comments or source timestamps."""
import argparse
from pathlib import Path, PurePosixPath
import stat
import zipfile


def repack(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")
    with zipfile.ZipFile(source) as original:
        members = [member for member in original.infolist() if not member.is_dir()]
        roots = {PurePosixPath(member.filename).parts[0] for member in members}
        if len(roots) != 1 or not members:
            raise ValueError("Expected a source ZIP with one top-level directory")
        entries = []
        names = set()
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or len(path.parts) < 2:
                raise ValueError("Unsafe source path")
            if any(part.startswith('.') or part in {'__MACOSX', '__pycache__'} for part in path.parts):
                raise ValueError("Hidden files or cache files require review before packaging")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Symbolic links require review before packaging")
            name = 'submission/' + str(PurePosixPath(*path.parts[1:]))
            if name in names:
                raise ValueError("Duplicate source path")
            names.add(name)
            mode = 0o755 if (member.external_attr >> 16) & 0o111 else 0o644
            entries.append((name, mode, original.read(member)))
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for name, mode, data in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, data)
    with zipfile.ZipFile(destination) as output:
        if output.comment or output.testzip() is not None:
            raise RuntimeError("Archive verification failed")
        for name, mode, data in entries:
            info = output.getinfo(name)
            if info.comment or info.extra or output.read(name) != data:
                raise RuntimeError("Content or metadata verification failed")
    print(f"Created {destination} with {len(entries)} files; contents preserved, metadata normalized.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    repack(args.source, args.destination)
