#!/bin/bash

for file in `find $1 -name "*.pth" -type f`;do 
    if [[ "$file" =~ .*"20000".* || "$file" =~ .*"250-0".* || "$file" =~ .*"best".* ]]; then
        echo "Skip $file"
        continue
    fi
    echo "delete $file"
done

#rm iter_1* && rm iter_4* && rm iter_6* && rm iter_8* && rm iter_20* && rm iter_22* && rm iter_24*