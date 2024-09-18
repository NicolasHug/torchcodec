#!/bin/bash

set -eux

source packaging/helpers.sh

wheel_path=$(pwd)/$(find dist -type f -name "*.whl")
echo "Wheel content:"
unzip -l $wheel_path

unamestr=$(uname)
if [[ "$unamestr" == 'Linux' ]]; then
    ext="so"
elif [[ "$unamestr" == 'Darwin' ]]; then
    ext="dylib"
else
    echo "Unknown operating system: $unamestr"
    exit 1
fi

for ffmpeg_major_version in 4 5 6 7; do
    assert_in_wheel $wheel_path torchcodec/libtorchcodec${ffmpeg_major_version}.${ext}
done
assert_not_in_wheel $wheel_path libtorchcodec.${ext}

for ffmpeg_ext in libavcodec.${ext} libavfilter.${ext} libavformat.${ext} libavutil.${ext} libavdevice.${ext} ; do
    assert_not_in_wheel $wheel_path $ffmpeg_ext
done

assert_not_in_wheel $wheel_path "^test"
assert_not_in_wheel $wheel_path "^doc"
assert_not_in_wheel $wheel_path "^benchmarks"
assert_not_in_wheel $wheel_path "^packaging"

if [[ "$unamestr" == 'Linux' ]]; then
    # See invoked python script below for details about this check.
    extracted_wheel_dir=$(mktemp -d)
    unzip -q $wheel_path -d $extracted_wheel_dir
    symbols_matches=$(find $extracted_wheel_dir | grep ".so$" | xargs objdump --syms | grep GLIBCXX_3.4.)
    python packaging/check_glibcxx.py "$symbols_matches"

    echo "ls dist"
    ls dist

    old="linux_x86_64"
    new="manylinux_2_17_x86_64.manylinux2014_x86_64"
    echo "Replacing ${old} with ${new} in wheel name"
    mv dist/*${old}*.whl $(echo dist/*${old}*.whl | sed "s/${old}/${new}/")
fi

echo "ls dist"
ls dist
