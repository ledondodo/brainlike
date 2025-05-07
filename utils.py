import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import explained_variance_score

def correlation_score(y_true, y_pred):
    return np.array([np.corrcoef(y_true[:,neuron], y_pred[:,neuron])[0,1] for neuron in range(y_pred.shape[1])])

def generate_test_data(n_samples=288, n_neurons=168, n_layers=6):
    """Génère des données de test aléatoires pour tester la visualisation.
    
    Args:
        n_samples (int): Nombre d'échantillons
        n_neurons (int): Nombre de neurones
        n_layers (int): Nombre de couches à simuler
        
    Returns:
        tuple: (predictions, spikes, keys)
    """
    # Générer les spikes aléatoires
    spikes = np.random.normal(0, 1, (n_samples, n_neurons))
    
    # Générer les prédictions pour chaque couche
    predictions = {}
    keys = {}
    
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    
    for i, layer in enumerate(layer_names[:n_layers]):
        # Générer des prédictions avec une corrélation décroissante avec les spikes
        correlation = 0.8 - (i * 0.1)  # La corrélation diminue avec la profondeur
        noise = np.random.normal(0, 1, (n_samples, n_neurons))
        pred = correlation * spikes + np.sqrt(1 - correlation**2) * noise
        predictions[layer] = pred
        
        # Générer des prédictions aléatoires pour la comparaison
        rand_pred = np.random.normal(0, 1, (n_samples, n_neurons))
        predictions[f'rand_{layer}'] = rand_pred
        
        keys[layer] = [f'rand_{layer}']
    
    return predictions, spikes, keys

def evaluate_super_compact(predictions, spikes, keys, bins=30, ev_range=(-2.5,1), output_file='comparison_layers.png'):
    evaluated_predictions = {k: v for k,v in predictions.items() if k in keys}
    
    # Créer une figure avec deux subplots côte à côte
    fig, (ax_ev, ax_corr) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Couleurs pour les différentes couches
    colors = plt.cm.viridis(np.linspace(0, 1, len(evaluated_predictions)))
    
    for i, (key, pred) in enumerate(evaluated_predictions.items()):
        ev = explained_variance_score(spikes, pred, multioutput='raw_values')
        corr = correlation_score(spikes, pred)
        
        print(f"--- {key} ---")
        print("Mean EV:", ev.mean())
        print("Mean Pearson correlation:", corr.mean())
        
        # Graphique pour la variance expliquée
        ax_ev.hist(ev, bins=bins, range=ev_range, alpha=0.3, 
                label=f'{key}', color=colors[i])
        
        # Graphique pour la corrélation
        ax_corr.hist(corr, bins=bins, range=(-0.5,1), alpha=0.3,
                label=f'{key}', color=colors[i])
    
    # Configurer le subplot EV
    ax_ev.set_title('Variance Expliquée par Couche')
    ax_ev.set_xlabel('Variance Expliquée')
    ax_ev.set_ylabel('Nombre de neurones')
    ax_ev.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configurer le subplot Corrélation
    ax_corr.set_title('Corrélation par Couche')
    ax_corr.set_xlabel('Coefficient de Corrélation')
    ax_corr.set_ylabel('Nombre de neurones')
    ax_corr.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Graphique sauvegardé dans {output_file}") 