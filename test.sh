cat texperiments.csv\
    | iconv -f cp932 > texperiments-utf.csv
cat drugs.csv\
    | iconv -f cp932 > drugs-utf.csv
cat panss.csv\
    | iconv -f cp932 > panss-utf.csv
# python -m cProfile -o prof dbfilter/mergedrug.py texperiments-utf.csv drugs-utf.csv 
# cat texperiments-utf.csv\
#     | python dbfilter/mergedrug.py drugs-utf.csv\
#     | python dbfilter/mergepsycho.py panss-utf.csv
cat texperiments-utf.csv\
    | python dbfilter/mergedrug.py drugs-utf.csv > with_drugs.csv

python dbfilter/mergepsycho.py panss-utf.csv with_drugs.csv > result1
cat with_drugs.csv\
    | python dbfilter/mergepsycho.py panss-utf.csv > result2
diff result1 result2

wc result1
wc result2
rm result1
rm result2
rm panss-utf.csv
rm drugs-utf.csv
rm texperiments-utf.csv
rm with_drugs.csv
