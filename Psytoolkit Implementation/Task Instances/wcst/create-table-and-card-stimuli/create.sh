#!/bin/bash

# wcst.svg has red shapes.  In order to make it green, blue, yellow,
# you just replace #ff0000 through the correct color name:
# green  = #00ff00
# blue   = #0000ff
# yellow = #ffff00

# ----------------------------------------------------------------------

# first create 4 wcsts

cp wcst.svg wcst.red.svg
sed s/#ff0000/#00ff00/g wcst.svg > wcst.green.svg
sed s/#ff0000/#0000ff/g wcst.svg > wcst.blue.svg
sed s/#ff0000/#ffff00/g wcst.svg > wcst.yellow.svg

# ----------------------------------------------------------------------

# first export all ids

for colorname in red green blue yellow
do

    for i in g2880 g2886 g2873 g2891 g2899 g2910 g2904 g2895 g2917 g2955 g2950 g2946 g2942 g2924 g2931 g2937
    do
	inkscape -i $i -t wcst.$colorname.svg
    done

    for ((i=1;i<5;i++))
    do
	mv circle$i.png circle$i$colorname.png
	mv star$i.png star$i$colorname.png
	mv cross$i.png cross$i$colorname.png
	mv triangle$i.png triangle$i$colorname.png
    done
    
done

# now remove tmp svg files
mv wcst.*.svg /tmp
