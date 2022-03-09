import argparse

parser = argparse.ArgumentParser(description='sasa')

parser.add_argument('--sasa', type=str, default='aaa', help='MLFlow tracking URI')
parser.add_argument('--saso', type=str, default='bbb', help='MLFlow tracking URI')

args = parser.parse_args()

for key in list(args.__dict__.keys()):
    print(getattr(args,key))
setattr(args, 'susu', 'ccc')
print(list(args.__dict__.keys()))