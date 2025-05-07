import numpy as np
import torch
import matplotlib.pyplot as plt
from fusion_model import MultiModalSpikePredictor, prepare_data_for_unified_model, train_unified_model

from utils import load_it_data

# Supposons que ces variables soient déjà définies dans votre environnement
# (stimulus_train, stimulus_val, object_train, object_val, spikes_train, spikes_val, num_classes)
# Sinon, vous devrez les charger à partir de vos données

def visualize_results(history, unified_predictions, obj_only_predictions, act_only_predictions, val_targets):
    """
    Visualise l'historique d'entraînement et compare les prédictions de différentes sources.
    """
    # Tracer l'historique d'entraînement
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_ev'], label='Variance Expliquée')
    plt.title('Variance Expliquée')
    plt.xlabel('Epochs')
    plt.ylabel('EV Score')
    plt.legend()
    
    # Histogramme des variances expliquées par neurone pour chaque méthode
    from sklearn.metrics import explained_variance_score
    
    unified_ev = np.array([explained_variance_score(val_targets[:, i], unified_predictions[:, i]) 
                          for i in range(val_targets.shape[1])])
    obj_only_ev = np.array([explained_variance_score(val_targets[:, i], obj_only_predictions[:, i]) 
                           for i in range(val_targets.shape[1])])
    act_only_ev = np.array([explained_variance_score(val_targets[:, i], act_only_predictions[:, i]) 
                           for i in range(val_targets.shape[1])])
    
    # Trouver les neurones où le modèle unifié est meilleur
    unified_better_than_both = np.logical_and(unified_ev > obj_only_ev, unified_ev > act_only_ev)
    
    plt.subplot(1, 3, 3)
    plt.hist([unified_ev, obj_only_ev, act_only_ev], bins=20, alpha=0.6, 
             label=['Unifié', 'Objets seulement', 'Activations seulement'])
    plt.title(f'Distribution des EV par Neurone\n{np.sum(unified_better_than_both)} neurones mieux prédits par le modèle unifié')
    plt.xlabel('Variance Expliquée')
    plt.ylabel('Nombre de Neurones')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Comparaison détaillée pour quelques neurones
    n_neurons_to_plot = min(5, val_targets.shape[1])
    plt.figure(figsize=(15, n_neurons_to_plot * 3))
    
    # Sélectionner les neurones où le modèle unifié est le meilleur
    if np.any(unified_better_than_both):
        best_neurons_idx = np.where(unified_better_than_both)[0][:n_neurons_to_plot]
    else:
        # Si aucun n'est meilleur, prendre les premiers
        best_neurons_idx = range(n_neurons_to_plot)
    
    for i, neuron_idx in enumerate(best_neurons_idx):
        plt.subplot(n_neurons_to_plot, 1, i+1)
        
        sample_indices = range(len(val_targets))
        plt.plot(sample_indices, val_targets[:, neuron_idx], 'k-', label='Vrai')
        plt.plot(sample_indices, unified_predictions[:, neuron_idx], 'r-', label='Unifié')
        plt.plot(sample_indices, obj_only_predictions[:, neuron_idx], 'g-', label='Objets')
        plt.plot(sample_indices, act_only_predictions[:, neuron_idx], 'b-', label='Activations')
        
        plt.title(f'Neurone {neuron_idx} - EV: Unifié={unified_ev[neuron_idx]:.3f}, '
                  f'Objets={obj_only_ev[neuron_idx]:.3f}, Activations={act_only_ev[neuron_idx]:.3f}')
        plt.xlabel('Échantillon')
        plt.ylabel('Activité du Spike')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Définir l'appareil de calcul
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    
    # Charger les données (à remplacer par votre propre chargement de données)
    stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data('.')
    num_classes = len(set(objects_train))

    # Préparer les données pour le modèle unifié
    print("Préparation des données et extraction des activations de la couche 4...")
    train_dataset, val_dataset = prepare_data_for_unified_model(
        stimulus_train, stimulus_val, objects_train, objects_val, spikes_train, spikes_val
    )
    
    # Obtenir la dimension des activations de la couche
    layer_activation_dim = train_dataset.layer_activations.shape[1]
    spike_dim = train_dataset.spikes.shape[1]
    
    print(f"Dimensions - Activations: {layer_activation_dim}, Spikes: {spike_dim}")
    
    # Initialiser le modèle
    model = MultiModalSpikePredictor(
        num_classes=num_classes,
        layer_activation_dim=layer_activation_dim,
        spike_dim=spike_dim
    ).to(device)
    
    # Entraîner le modèle
    print("Début de l'entraînement du modèle unifié...")
    history = train_unified_model(
        model, train_dataset, val_dataset, 
        num_epochs=20, batch_size=32, learning_rate=0.001, device=device
    )
    
    # Évaluer le modèle
    print("Évaluation du modèle...")
    model.eval()
    with torch.no_grad():
        # Obtenir les prédictions du modèle unifié et des branches individuelles
        val_objects = val_dataset.objects
        val_activations = val_dataset.layer_activations
        val_targets = val_dataset.spikes.cpu().numpy()
        
        unified_predictions = model(val_objects, val_activations).cpu().numpy()
        obj_only_predictions = model.forward_objects_only(val_objects).cpu().numpy()
        act_only_predictions = model.forward_activations_only(val_activations).cpu().numpy()
    
    # Visualiser les résultats
    visualize_results(history, unified_predictions, obj_only_predictions, act_only_predictions, val_targets)
    
    print("Terminé!")

if __name__ == "__main__":
    main() 