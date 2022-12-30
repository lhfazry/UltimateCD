#!/bin/bash

for file in `find $1 -name "*.pth" -type f`;do 
    if [[ "$file" =~ .*"20k".* ]]; then
        if [[ "$file" =~ .*"20000".* || "$file" =~ .*"best".* ]]; then
            echo "Skip $file"
            continue
        fi
    fi

    if [[ "$file" =~ .*"25k".* ]]; then
        if [[ "$file" =~ .*"25000".* || "$file" =~ .*"best".* ]]; then
            echo "Skip $file"
            continue
        fi
    fi

    if [[ "$file" =~ .*"50k".* ]]; then
        if [[ "$file" =~ .*"50000".* || "$file" =~ .*"best".* ]]; then
            echo "Skip $file"
            continue
        fi
    fi

    # Skip the target file in latest.pth
    parent_dir=`dirname $file`
    target_latest_pth=`readlink -f $parent_dir/latest.pth`
    echo "Target latest.pth: $target_latest_pth"

    echo "Delete $file"
done