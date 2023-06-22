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

# Environments

+ python 3.8
+ pytorch 1.70

# Dataset

1. Task1
2. Task2
3. Task3

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
task3.ipynb
```
