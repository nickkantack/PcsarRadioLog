import os
import json


def main():

    labeled_data = []
    unlabeled_count = 0
    for filename in os.listdir("../backend/data"):
        if filename.endswith(".txt"):
            with open(f"../backend/data/{filename}", "r") as file:
                properties = json.loads(file.read())
                if "label" in properties:
                    labeled_data.append(properties)
                else:
                    unlabeled_count += 1

    print(f"There are {len(labeled_data)} files with labels, which is {100 * len(labeled_data) / unlabeled_count:.1f}% of them")

    for i in range(5):
        print(labeled_data[i])

    print("Done")


if __name__ == "__main__":
    main()