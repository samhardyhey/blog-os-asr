#!/bin/bash
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
extensions=$(cat $dir/extensions.txt)

# Install each extension using the code command
for extension in $extensions; do
    code --install-extension $extension
done