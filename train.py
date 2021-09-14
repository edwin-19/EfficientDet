import argparse
import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/archive/train_zip/train')
    parser.add_argument('--test_dir', default='data/archive/test_zip/test')
    
    args = parser.parse_args()
    
    train_df = preprocess.convert_annots(args.train_dir)
    test_df = preprocess.convert_annots(args.test_dir)
    print(test_df)