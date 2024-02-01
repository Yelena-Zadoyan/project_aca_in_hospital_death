import argparse
from pipeline import Pipeline


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help='Path to the dataset')
parser.add_argument('--test', action='store_true', help='Run in testing mode')
args = parser.parse_args()
print(args.data_path, args.test)
pipeline = Pipeline(args.data_path, test=args.test)
pipeline.run()
