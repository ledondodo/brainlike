from utils import generate_test_data, evaluate_super_compact

# Générer les données de test
predictions, spikes, keys = generate_test_data()

# Visualiser les résultats et sauvegarder dans un fichier
evaluate_super_compact(predictions, spikes, keys, ev_range=(-1,1), output_file='comparison_layers_test.png') 