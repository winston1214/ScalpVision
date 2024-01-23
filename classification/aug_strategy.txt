import pandas as pd
import random
import csv
import argparse

def cal_pick_ratio(df, column):
    """
    Calculates the pick ratio of a given column in a dataframe.
    The pick ratio is the proportion of rows in the dataframe where
    the value of the specified column is not null.

    Args:
    df (pd.DataFrame): The dataframe to calculate the pick ratio for.
    column (str): The column name to calculate the pick ratio for.

    Returns:
    list: A list of pick ratios for each unique value in the column.
    """
    label = f'value_{column}'
    s = sum(1 / df[label].value_counts(normalize=True))
    dst = 1 / df[label].value_counts(normalize=True) / s
    return [dst[i] for i in range(4)]

def pick_rd_tgt(df, c1, c2, c3, src_img_name):
    """
    Selects a random target image that meets specified criteria and is
    different from the source image.

    Args:
    df (pd.DataFrame): The dataframe to select from.
    c1, c2, c3 (int): The values for the three columns to match.
    src_img_name (str): The name of the source image to avoid.

    Returns:
    pd.Series: A randomly selected row from the dataframe.
    """
    sp = df[(df.value_1 == c1) & (df.value_2 == c2) & (df.value_3 == c3)].sample(n=1)
    if sp.iloc[0]['img_name'] == src_img_name:
        return pick_rd_tgt(df, c1, c2, c3, src_img_name)
    else:
        return sp

def main(input_file, output_file, sample_size, augmentation_size):
    try:
        df = pd.read_csv(input_file)

        og_df = df.copy()
        pair = []
        for _ in range(augmentation_size):
            sample_df = og_df.sample(n=sample_size)
            c1_dst = cal_pick_ratio(df, 1)
            c2_dst = cal_pick_ratio(df, 2)
            c3_dst = cal_pick_ratio(df, 3)

            for i in range(sample_size):
                c1, c2, c3 = random.choices([0, 1, 2, 3], weights=c1_dst, k=1)[0], random.choices([0, 1, 2, 3], weights=c2_dst, k=1)[0], random.choices([0, 1, 2, 3], weights=c3_dst, k=1)[0]
                img_name = sample_df.iloc[i]['img_name'].split('.jpg')[0] + f'_{c1}{c2}{c3}.jpg'
                df = pd.concat([df, pd.DataFrame([{ 'value_1': c1, 'value_2': c2, 'value_3': c3, 'img_name': img_name }])], ignore_index=True)
                pair.append({"src_img_name": sample_df.iloc[i]['img_name'],
                            "target_img_name": pick_rd_tgt(og_df, c1, c2, c3, sample_df.iloc[i]['img_name']).iloc[0]['img_name'],
                            "src_label": f"{sample_df.iloc[i]['value_1']}{sample_df.iloc[i]['value_2']}{sample_df.iloc[i]['value_3']}",
                            "target_label": f'{c1}{c2}{c3}'})

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['src_img_name', 'target_img_name', 'src_label', 'target_label'])
            writer.writeheader()
            writer.writerows(pair)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Augmentation for Image Labeling')
    parser.add_argument('--input_file', type=str, help='Input CSV file path')
    parser.add_argument('--output_file', type=str, help='Output CSV file path')
    parser.add_argument('--sample_size', type=int, default=1000, help='Sample size for each augmentation iteration')
    parser.add_argument('--augmentation_size', type=int, default=50, help='Number of augmentation iterations')

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.sample_size, args.augmentation_size)