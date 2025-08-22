from packages.dataset_processing import *
from packages.neural_network import *
from packages.nlp import *
import csv


def create_datasets():
    dataset = pd.read_csv('./datasets/Dataset.csv')
    source_lang = "fr"
    target_lang = "en"
    initial_analysis(dataset)
    print("\n")
    extract_distinct_value_list(dataset)
    print("\n")
    # histogram(dataset)
    # countplot(dataset)
    # translate(dataset, source_lang, target_lang)

    df = pd.read_csv('./datasets/Translated_Dataset.csv')
    balanced_df = balance_classes(df, 150)
    balanced_df.to_csv('./datasets/Balanced_Dataset.csv', index=False)
    initial_analysis(balanced_df)
    print("\n")
    extract_distinct_value_list(balanced_df)
    print("\n")
    text_to_numeric_mappings = convertToNumeric(balanced_df)
    # showCorrelations(balanced_df)

    return text_to_numeric_mappings


def prompt_user(keywords, found_keywords, keyword_contexts):
    scores = []

    outdoor_time = get_valid_input("How long does your cat spend outdoors (0-5): ")
    time_spent_with_cat = get_valid_input("How much time do you spend with your cat (0-5): ")

    scores.append(int(outdoor_time))
    scores.append(int(time_spent_with_cat))

    for keyword in keywords:
        if keyword in found_keywords:
            negated = False

            for context in keyword_contexts[keyword]:
                if "Negated" in context:
                    negated = True
                    break

            if negated:
                print(f"{keyword.capitalize()} is negated, setting score to 0.")
                scores.append(0)
            else:
                score = get_valid_input(
                    f"How would you rank the level of {keyword} on a scale from 0 to 5 (0 being least and 5 being most): ")
                scores.append(int(score))
        else:
            scores.append(0)

    return np.array(scores)


def get_valid_input(prompt):
    while True:
        try:
            user_input = int(input(prompt))
            if 0 <= user_input <= 5:
                return user_input
            else:
                print("Invalid input. Please enter a number between 0 and 5.")
        except ValueError:
            print("Invalid input. Please enter a valid integer between 0 and 5.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <text|file_path>")
        sys.exit(1)

    input_data = sys.argv[1]
    if os.path.isfile(input_data):
        text = text_from_file(input_data)
    else:
        text = input_data

    print("Original Text:")
    print("-" * 50)
    print(text)

    keywords = ["shy", "calm", "scared", "intelligent", "vigilant", "persistent", "affectionate", "friendly",
                "solitary",
                "rough", "dominant", "aggressive", "impulsive", "predictable", "distracted", "wealthy", "birds",
                "rodents"]

    overall_contexts = {}
    keyword_contexts = find_keywords(text, keywords)
    found_keywords = set()
    print("\nKeyword Matches and Context:")
    print("-" * 50)
    for keyword, contexts in keyword_contexts.items():
        print(f"{keyword.capitalize()}:")
        found_keywords.add(keyword)
        overall_contexts[keyword] = contexts
        for context in contexts:
            print(f"  - {context}")
        # print(generate_sentence_with_keyword(str(keyword)))

    # stylometry(text)

    # generate_alternatives(text)

    while len(found_keywords) < 6:
        user_description = input(
            f"You need to use at least 6 different keywords. Currently, you have {len(found_keywords)} "". Please enter a new description: ")

        keyword_contexts = find_keywords(user_description, keywords)

        for keyword, contexts in keyword_contexts.items():
            print(f"{keyword.capitalize()}:")
            found_keywords.add(keyword)
            overall_contexts[keyword] = contexts
            for context in contexts:
                print(f"  - {context}")

    rankings = prompt_user(keywords, found_keywords, overall_contexts)

    return rankings


def gen(ext, obs, shy, calm, scared, intelligent, vigilant, persistent, affectionate, friendly, solitary, rough,
        dominant, aggressive, impulsive, predictable, distracted, abundance, predBirds, predMamm):
    TEMPLATES = [
        f"This cat {ext} extroverted, {obs} observant, {shy} shy, and {calm} calm. Although it {scared} scared, it {intelligent} intelligent and it {vigilant} vigilant. It {persistent} persistent, {affectionate} affectionate, and {friendly} friendly, though {solitary} solitary. Its behavior {rough} rough, showing that it {dominant} dominant and {aggressive} aggressive tendencies, yet it {impulsive} impulsive and {predictable} predictable. The cat {distracted} distracted and it {abundance} abundance when interacting with predators, so it {predBirds} wary of birds and {predMamm} wary of mammals.",

        f"Known that it {ext} extroverted and {obs} observant, this cat {shy} shy and {calm} calm. It {scared} scared but it {intelligent} intelligent and {vigilant} vigilant. It {persistent} persistent and {affectionate} affectionate and because of its nature, it {friendly} friendly and it {solitary} solitary. Despite it {rough} rough, it {dominant} dominant and it {aggressive} aggressive while it {impulsive} impulsive and {predictable} predictable. It {distracted} distracted and {abundance} abundance when reacting to predators, {predBirds} wary of birds and {predMamm} wary of mammals.",

        f"With a personality that {ext} extroverted and {obs} observant, this cat {shy} shy but {calm} calm. It faces challenges when it {scared} scared yet {intelligent} intelligent and {vigilant} vigilant. The cat {persistent} persistence and {affectionate} affectionate and it {friendly} friendly despite it {solitary} solitary. It {rough} roughly and {dominant} dominant and {aggressive} aggressive tendencies, showing a balance because it {impulsive} impulsive and {predictable} predictable. The cat {distracted} distracted and demonstrates that it {abundance} abundance while it {predBirds} wary of predatory birds and {predMamm} wary of mammals.",

        f"Despite it {shy} shy and {scared} scared, this cat's nature {ext} extroverted and {obs} observant. It {calm} calm, {intelligent} intelligent, and {vigilant} vigilant. The cat {persistent} persistent, {affectionate} affectionate, and {friendly} friendly, it {solitary} solitary. It {rough} rough and it {dominant} dominant and {aggressive} aggressive. Its behavior {impulsive} impulsive yet {predictable} predictable, so it {distracted} distracted. It {abundance} abundance and {predBirds} wary of birds and {predMamm} wary of mammals."
    ]

    return random.choice(TEMPLATES)


def getLevelAtribute(rank):
    if (rank == 0):
        return random.choice(["is not showing", "does not display", "lacks any display of", "fails to exhibit"])
    elif (rank == 1):
        return random.choice(["is barely showing", "displays minimal", "has shown slight", "shows very little"])
    elif (rank == 2):
        return random.choice(
            ["shows little", "displays some", "has demonstrated a modest amount of", "is somewhat showing"])
    elif (rank == 3):
        return random.choice(
            ["shows moderate", "displays a fair amount of", "has shown moderate", "demonstrates moderate"])
    elif (rank == 4):
        return random.choice(
            ["is strongly showing", "displays a strong", "has significantly shown", "demonstrates a high level of"])
    elif (rank == 5):
        return random.choice(
            ["is extremely showing", "displays an intense", "has shown a very high level of", "demonstrates maximum"])
    else:
        return ""


def generateDescription(rankings, breed):
    if ("No breed" in breed or "Other" in breed or "Unknown breed" in breed):
        print("We can't recognize your cat type. Try another description.")
    else:
        print(gen(getLevelAtribute(rankings[0]), getLevelAtribute(rankings[1]), getLevelAtribute(rankings[2]),
                  getLevelAtribute(rankings[3]), getLevelAtribute(rankings[4]), getLevelAtribute(rankings[5]),
                  getLevelAtribute(rankings[6]), getLevelAtribute(rankings[7]), getLevelAtribute(rankings[8]),
                  getLevelAtribute(rankings[9]), getLevelAtribute(rankings[10]), getLevelAtribute(rankings[11]),
                  getLevelAtribute(rankings[12]), getLevelAtribute(rankings[13]), getLevelAtribute(rankings[14]),
                  getLevelAtribute(rankings[15]), getLevelAtribute(rankings[16]), getLevelAtribute(rankings[17]),
                  getLevelAtribute(rankings[18]), getLevelAtribute(rankings[19])))


def predictCatType(rankings):
    text_to_numeric_mappings = create_datasets()

    predicted_class = neural_network_training(rankings)
    for breed, value in text_to_numeric_mappings['Breed'].items():
        if value == predicted_class:
            if breed == 'BEN':
                return "Bengal"
            elif breed == 'BIR':
                return "Birman"
            elif breed == 'BRI':
                return "British Shorthair"

            elif breed == 'CHA':
                return "Chartreux"

            elif breed == 'EUR':
                return "European"
            elif breed == 'MCO':
                return "Maine coon"
            elif breed == 'PER':
                return "Persian"
            elif breed == 'RAG':
                return "Ragdoll"
            elif breed == 'SAV':
                return "Savannah"
            elif breed == 'SPH':
                return "Sphinx"
            elif breed == 'TUR':
                return "Turkish angora"


def compare_cats(cat1_ranks, cat2_ranks, cat2_name):
    attributes = [
        "extroverted", "observant", "shy", "calm", "scared",
        "intelligent", "vigilant", "persistent", "affectionate", "friendly",
        "solitary", "rough", "dominant", "aggressive", "impulsive",
        "predictable", "distracted", "abundance", "wary of birds", "wary of mammals"
    ]
    # cat2_name=str(predictCatType(cat2_ranks))+" Cat "
    comparisons = []

    for i in range(0, len(attributes) - 1):
        attribute = attributes[i]
        rank1 = int(cat1_ranks[i])
        rank2 = int(cat2_ranks[i])
        if int(rank1) < int(rank2):
            comparisons.append(f"Your cat is less {attribute} than {cat2_name}.")
        elif int(rank1) > int(rank2):
            comparisons.append(f"Your cat is more {attribute} than {cat2_name}.")
        else:
            comparisons.append(f"Your cat and {cat2_name} are equally {attribute}.")

    print("\n".join(comparisons))


def extract_columns_from_csv(file_path):
    selected_columns = list(range(3, 28))
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        data = [[row[i] for i in selected_columns] for row in reader]
    return data


if __name__ == '__main__':
    rankings = main()

    text_to_numeric_mappings = create_datasets()

    predicted_class = neural_network_training(rankings)

    extracted_data = random.choice(extract_columns_from_csv("./datasets/Numeric_Dataset.csv")[8:])
    cat_name = random.choice(extract_columns_from_csv("./datasets/Numeric_Dataset.csv")[4])
    print("%%%%%%%%%%%%", type(cat_name))
    cat_name = int(cat_name)
    name_cat = str()
    for breed, value in text_to_numeric_mappings['Breed'].items():
        if value == cat_name:
            if breed == 'BEN':
                name_cat = "Bengal"
            elif breed == 'BIR':
                name_cat = "Birman"
            elif breed == 'BRI':
                name_cat = "British Shorthair"
            elif breed == 'CHA':
                name_cat = "Chartreux"
            elif breed == 'EUR':
                name_cat = "European"
            elif breed == 'MCO':
                name_cat = "Maine coon"
            elif breed == 'PER':
                name_cat = "Persian"
            elif breed == 'RAG':
                name_cat = "Ragdoll"
            elif breed == 'SAV':
                name_cat = "Savannah"
            elif breed == 'SPH':
                name_cat = "Sphinx"
            else:
                name_cat = "Turkish angora"

    name_cat += " Cat"
    for breed, value in text_to_numeric_mappings['Breed'].items():
        if value == predicted_class:
            if breed == 'BEN':
                print("The cat is Bengal.", end=" ")
                generateDescription(rankings, "Bengal")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'BIR':
                print("The cat is Birman. ", end=" ")
                generateDescription(rankings, "Birman")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'BRI':
                print("The cat is British Shorthair. ", end=" ")
                generateDescription(rankings, "British Shorthair")
                compare_cats(rankings, extracted_data, name_cat)

            elif breed == 'CHA':
                print("The cat is Chartreux. ", end=" ")
                generateDescription(rankings, "Chartreux")
                compare_cats(rankings, extracted_data, name_cat)

            elif breed == 'EUR':
                print("The cat is European. ", end=" ")
                generateDescription(rankings, "European")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'MCO':
                print("The cat is Maine coon. ", end=" ")
                generateDescription(rankings, "Maine coon")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'PER':
                print("The cat is Persian. ", end=" ")
                generateDescription(rankings, "Persian")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'RAG':
                print("The cat is Ragdoll. ", end=" ")
                generateDescription(rankings, "Ragdoll")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'SAV':
                print("The cat is Savannah. ", end=" ")
                generateDescription(rankings, "Savannah")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'SPH':
                print("The cat is Sphinx. ", end=" ")
                generateDescription(rankings, "Sphinx")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'TUR':
                print("The cat is Turkish angora. ", end=" ")
                generateDescription(rankings, "Turkish angora")
                compare_cats(rankings, extracted_data, name_cat)
            elif breed == 'NR':
                print("No breed")
                generateDescription(rankings, "No breed")
            elif breed == 'Other':
                print("Other")
                generateDescription(rankings, "Other")
            else:
                print("Unknown breed")
                generateDescription(rankings, "Unknown breed")
