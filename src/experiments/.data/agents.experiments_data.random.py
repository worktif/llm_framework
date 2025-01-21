import os
import random

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.files import save_to_json
from src.utils.objects import deep_join
from src.utils.strings import generate_random_word

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Define the output folder for experiment data
output_folder_experiment_data = f"{current_dir}"

# Fibonacci sequence for context window sizes
fibonacci = [1, 2, 3, 5, 8, 13, 21]

# Initialize agent models
agent_models = {
    "A1": SentenceTransformer('all-MiniLM-L6-v2'),
    "A2": SentenceTransformer('paraphrase-MiniLM-L6-v2'),
    "A3": SentenceTransformer('paraphrase-albert-small-v2'),
    "A4": SentenceTransformer('all-MiniLM-L12-v2'),
    "A5": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),
}

# Initialize queries and responses (currently empty)
queries = [
    "What is the weather today?",
    "What can I do on a sunny day?",
    "What are some popular outdoor activities?",
    "How do I prepare for a hike?",
    "What kind of gear do I need for hiking?",
    "Where can I buy hiking gear?",
    "What are the best hiking trails nearby?",
    "How can I find detailed trail maps?",
    "What are the safety tips for hiking?",
    "What should I do if I encounter wildlife on a trail?",
    "What precautions should I take to avoid getting lost?",
    "How do I use a compass for navigation?",
    "Can I rely on my phone's GPS for hiking?",
    "What are some good apps for hiking and navigation?",
    "How can I track my fitness while hiking?",
    "What are the health benefits of hiking?",
    "What is the average calorie burn during a hike?",
    "What are some healthy snacks to bring on a hike?",
    "How much water should I bring for a day hike?",
    "What is the best way to carry water while hiking?",
    "How do I avoid dehydration during physical activities?",
    "What are the symptoms of dehydration?",
    "How do I treat dehydration if it occurs?",
    "What are the long-term effects of chronic dehydration?",
    "What are the best practices for staying hydrated daily?",
    "What is the recommended daily water intake?",
    "How does water intake vary with physical activity?",
    "How does hydration affect mental performance?",
    "What foods can help with hydration?",
    "What is the role of electrolytes in hydration?",
    "What are some natural sources of electrolytes?",
    "How do sports drinks compare to water for hydration?",
    "What are the potential downsides of sugary sports drinks?",
    "What are the benefits of making homemade electrolyte drinks?",
    "How do I prepare a homemade electrolyte drink?",
    "What are the best times to consume electrolyte drinks?",
    "How do electrolyte needs vary with age?",
    "What are the signs of an electrolyte imbalance?",
    "What are the risks of overhydration?",
    "How do I balance hydration and electrolyte levels?",
    "What are the signs of a well-hydrated body?",
    "How does hydration impact skin health?",
    "What is the connection between hydration and energy levels?",
    "What are some hydration myths?",
    "How do caffeinated beverages affect hydration?",
    "What is the role of hydration in sports performance?",
    "How do professional athletes manage hydration?",
    "What tools do athletes use to monitor hydration?",
    "How do extreme weather conditions affect hydration needs?",
    "What are the hydration strategies for desert climates?"
]
random.shuffle(queries)
random.shuffle(queries)
random.shuffle(queries)

responses = [
    "The weather today is sunny.",
    "On a sunny day, you can go hiking, have a picnic, or visit the beach.",
    "Popular outdoor activities include hiking, biking, and kayaking.",
    "To prepare for a hike, wear comfortable clothes, sturdy shoes, and pack water and snacks.",
    "Essential hiking gear includes a backpack, water bottle, first-aid kit, and trail map.",
    "You can buy hiking gear at outdoor stores like REI or Decathlon.",
    "Some of the best trails nearby include Green Valley Trail and Sunset Peak.",
    "You can find detailed trail maps on apps like AllTrails or websites like hikingproject.com.",
    "Key hiking safety tips include staying on marked trails, carrying enough water, and informing someone of your plans.",
    "If you encounter wildlife, stay calm, do not approach, and slowly back away.",
    "To avoid getting lost, stick to the trail, carry a map and compass, and download offline maps.",
    "To use a compass, align it with your map and follow the marked direction.",
    "Phone GPS is helpful but can fail due to low battery or signal loss.",
    "Good apps for hiking include AllTrails, Gaia GPS, and Komoot.",
    "To track fitness while hiking, you can use fitness trackers like Fitbit or Garmin.",
    "Hiking improves cardiovascular health, strengthens muscles, and reduces stress.",
    "An average person burns 400-700 calories per hour hiking.",
    "Healthy snacks include nuts, dried fruits, and granola bars.",
    "Bring at least 2 liters of water for a day hike.",
    "Hydration packs or collapsible bottles are convenient for carrying water.",
    "Avoid dehydration by drinking small amounts of water regularly.",
    "Symptoms of dehydration include dry mouth, fatigue, and dark urine.",
    "To treat dehydration, drink water slowly and consider an electrolyte drink.",
    "Chronic dehydration can lead to kidney problems and low energy levels.",
    "Stay hydrated by drinking water throughout the day and eating hydrating foods.",
    "The recommended daily water intake is about 2-3 liters for adults.",
    "Increase water intake during physical activities to replace lost fluids.",
    "Good hydration supports better focus, memory, and overall mental performance.",
    "Foods like watermelon, cucumber, and oranges help with hydration.",
    "Electrolytes like sodium and potassium maintain fluid balance in the body.",
    "Natural sources of electrolytes include bananas, spinach, and coconut water.",
    "Sports drinks provide electrolytes but may contain excessive sugar.",
    "Sugary sports drinks can lead to weight gain and energy crashes.",
    "Homemade electrolyte drinks are cost-effective and customizable.",
    "Mix water, lemon juice, salt, and honey to make a homemade electrolyte drink.",
    "Consume electrolyte drinks before, during, and after intense exercise.",
    "Older adults may need more electrolytes to maintain balance.",
    "Signs of an electrolyte imbalance include fatigue, muscle cramps, and irregular heartbeat.",
    "Overhydration can cause nausea, confusion, and dangerously low sodium levels.",
    "Balance hydration by drinking water gradually and consuming electrolytes as needed.",
    "A well-hydrated body shows clear urine, good energy, and healthy skin.",
    "Proper hydration keeps skin supple and reduces dryness.",
    "Staying hydrated boosts energy and prevents fatigue.",
    "Common hydration myths include the need for 8 glasses daily for everyone.",
    "Caffeinated drinks can dehydrate in excess, but moderate intake is fine.",
    "Hydration improves endurance and recovery during sports.",
    "Professional athletes monitor hydration through sweat tests and weight tracking.",
    "Athletes use tools like smart bottles and hydration tracking apps.",
    "Hot, dry climates require more frequent hydration breaks.",
    "In desert climates, drink water consistently and wear sun-protective clothing."
]
random.shuffle(responses)
random.shuffle(responses)
random.shuffle(responses)

# System context for agents
system_context = ["""
    System Context:
        The system is a conversational AI model designed to provide clear, concise, and accurate information to user queries. The model should prioritize user understanding by maintaining relevance, logical consistency, and a natural flow of information throughout the dialogue. Responses should be informative yet succinct, addressing the user's query directly while avoiding unnecessary verbosity.
        The system is knowledgeable in various domains, including outdoor activities, hydration strategies, health, and fitness. It aims to guide the user through their inquiries by providing practical advice and actionable recommendations.
        Additionally, the system assumes that the user has basic knowledge of general topics and seeks to enhance their understanding or provide guidance on specific queries. The system is expected to stay on topic and adjust its responses based on the evolving context of the conversation.
"""]

# Organize agent data
agent_data = {
    agent: {
        "queries": queries,
        "responses": responses,
    } for agent in agent_models.keys()
}

# Save agent data to JSON
save_to_json(output_folder_experiment_data, "agent_data.random", agent_data)

def update_agent_context(agent, agent_context, current_query, current_response, window):
    """
    Update the agent's conversational context and calculate the deviation.

    Parameters:
    - agent: The agent identifier.
    - agent_context: The current context for the agent.
    - current_query: The user's query.
    - current_response: The agent's response.
    - window: The context window size.

    Returns:
    - deviation: The calculated deviation value.
    """
    H_prev = agent_context[-window:]

    # Generate random words for embedding demonstration
    random_words = " ".join(generate_random_word(random.randint(3, 8)) for _ in range(100))
    agent_context_step = [f"User: {current_query}", f"LLM: {current_response}"]

    # Create embeddings for the previous context
    previous_context = deep_join(H_prev)
    previous_context_embedding = agent_models[agent].encode(previous_context)

    # Update the current context
    H_prev.append(deep_join(agent_context_step))

    # Create embedding for the current step
    response_embedding = agent_models[agent].encode(deep_join(agent_context_step))

    # Calculate deviation (1 - cosine similarity)
    deviation = 2 * (1 - cosine_similarity([response_embedding], [previous_context_embedding])[0][0])
    agent_context.append(agent_context_step)

    return deviation

# Initialize agent contexts
agent_contexts = {
    agent: {
        experiment_window_size: [system_context] for experiment_window_size in fibonacci
    } for agent in agent_data.keys()
}

agent_deviations = {}

# Calculate deviations for each agent
for agent, data in agent_data.items():
    for experiment_window_size in fibonacci:
        deviations = []
        for query, response in zip(data["queries"], data["responses"]):
            current_response_embedding = agent_models[agent].encode(response)
            deviation = update_agent_context(agent, agent_contexts[agent][experiment_window_size], query, response, experiment_window_size)
            deviations.append(deviation)
        if agent not in agent_deviations:
            agent_deviations[agent] = {}
        agent_deviations[agent][experiment_window_size] = deviations

# Save deviations to JSON
save_to_json(output_folder_experiment_data, "agent_deviations.random", agent_deviations)