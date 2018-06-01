import pickle as pkl
import pandas as pd
if __name__ == '__main__':
    pkl.HIGHEST_PROTOCOL = 2
    df = pd.read_pickle(r"/home/samuel/dataset/eye-closed/dataset_B_FacialImages/train.pkl")
    print(df)
