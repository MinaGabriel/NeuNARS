# Builds NARS training data from the HotpotQA train split (see train_data.py). Resumable: if it stops for any reason,
# run the same command again and it continues where it left off, without redoing or duplicating a question.
#   python gen_hotpotqa.py --n 50                 a pilot
#   python gen_hotpotqa.py --n 1500               the full set (continues from the pilot)
#   python gen_hotpotqa.py --n 1500 --budget 15   stop early if the total cost reaches $15
from train_data import main

if __name__ == "__main__":
    main("HotpotQA")
