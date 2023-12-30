# Discerning the complementary and substitutive effects of generative AI on job skills fosters career sustainability

# Main Steps

1. Task1

   1. Collect expert annotations. (data/gpt_score.xlsx)
   2. Obtain the features of positions and skills. (models/best_model.pt)
   3. Aggregate these impacts to the occupational level. (results/gpt_impact.csv)
2. Task2

   1. Obtain the features of positions and skills. (models/2208_2212_model.pt, models/2212_2304_model.pt)
   2. Compute the similarity. (results/similarity_2208_2212.csv, results/2212_2304/csv)
3. Task3

   1. Obtain the Onet Skill and corresponding scores. (data/task3/Skills.xlsx, data/task4/Onet_triplet.tsv)
   2. Collect expert annotations. (data/task4/ONET_annotation_0821.xlsx)
   3. Obtain the features of positions and skills. (models/best_model.pt)
   4. Aggregate these impacts to the occupational level. (results/task4_gpt_impact_AVERAGE.csv)
4. Notice：
   Files in figures/ are stata data and codes for results 2.3 and 2.5.

# Environments

+ python 3.8
+ pytorch 1.13.0

# Running

Task1

```shell
python train.py task1
python task1.py 
```

Task2

```shell
python train.py task2_2208_2212
python task2.py task2_2208_2212

python train.py task2_2212_2304
python task2.py task2_2212_2304
```

Task3

```
python train.py task3
python task3.py task3
```
