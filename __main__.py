from dataset import EyeClosedDataset
from nets import EyeClosedNet

def main():
    left_eye = False
    dataset = EyeClosedDataset("/home/mtk/dataset/eye-closed/dataset_B_Eye_Images/",image_shape=(24,24),left_eye=left_eye)
    dataset.load_dataset()
    net = EyeClosedNet(dataset)
    net.train(left_eye=left_eye)

if __name__ == "__main__":
    main()
