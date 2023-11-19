import random

def generate_kiki_bouba_prompts(num_prompts=100):
    # Define categories of adjectives and entities
    adjectives = {
        "shape": ["rounded", "smooth", "curved", "spherical", "oval", "spikey", "jagged", "pointed", "angular", "sharp"],
        "size": ["large", "massive", "huge", "gigantic", "enormous", "small", "tiny", "miniature", "petite", "little"],
        "emotion": ["scary", "menacing", "fierce", "intimidating", "threatening", "peaceful", "gentle", "calm", "serene", "soothing"]
    }

    entities = {
        "natural": ["rock", "hill", "mountain", "cloud", "river", "cactus", "hedgehog", "volcano", "iceberg", "wave"],
        "living": ["dog", "cat", "lion", "puppy", "kitten", "bear", "elephant", "mouse", "fish", "bird"],
        "man_made": ["building", "sculpture", "vehicle", "bridge", "tower", "toy", "ball", "chair", "house", "machine"]
    }

    # Generate prompts
    prompts = []
    for _ in range(num_prompts):  # Generate 10 prompts
        adjective_category = random.choice(list(adjectives.keys()))
        entity_category = random.choice(list(entities.keys()))

        adjective = random.choice(adjectives[adjective_category])
        entity = random.choice(entities[entity_category])

        prompt = f"{adjective} {entity}"
        prompts.append(prompt)

    return prompts

# Generate and display the prompts
generated_prompts = generate_kiki_bouba_prompts()
generated_prompts