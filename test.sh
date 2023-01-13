cat texperiments.csv\
    | iconv -f cp932 > texperiments-utf.csv
cat drugs.csv\
    | iconv -f cp932 > drugs-utf.csv
cat panss.csv\
    | iconv -f cp932 > panss-utf.csv
cat drugs-utf.csv\
    | shrinkdrug > shrinked-drugs-utf.csv
cat texperiments-utf.csv\
    | mergedrug shrinked-drugs-utf.csv > with_drugs.csv

mergepsycho panss-utf.csv with_drugs.csv > result1
cat with_drugs.csv\
    | mergepsycho panss-utf.csv > result2
diff result1 result2

wc result1
wc result2
rm result1
rm result2
rm panss-utf.csv
rm drugs-utf.csv
rm texperiments-utf.csv
rm with_drugs.csv
rm shrinked_drugs-utf.csv
