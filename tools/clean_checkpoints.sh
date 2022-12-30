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

    echo "Delete $file"
done