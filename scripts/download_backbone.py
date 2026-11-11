"""Download the official DINOv2 ViT-B/14 backbone when explicitly invoked."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
SHA256 = '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path.home() / '.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth')
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists():
        if digest(destination) != SHA256:
            raise ValueError(f'Existing file has an unexpected SHA256: {destination}')
        print(f'Already verified: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.download')
    print(f'Downloading {URL}', flush=True)
    try:
        urllib.request.urlretrieve(URL, temporary)
        if digest(temporary) != SHA256:
            raise ValueError('Downloaded backbone failed SHA256 verification')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Verified: {destination}')


if __name__ == '__main__':
    main()
