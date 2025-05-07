import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import explained_variance_score
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class MultiModalSpikePredictor(nn.Module):
    """
    Un modèle qui combine les informations des classes d'objets et des activations
    de couches d'un ResNet pour prédire les spikes neuronaux.
    """
    def __init__(self, num_classes, layer_activation_dim=1000, embedding_dim=64, 
                 hidden_dim=128, fusion_dim=256, spike_dim=168, dropout_rate=0.2):
        super().__init__()
        
        # Branche 1: Pour les classes d'objets
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.obj_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.obj_fc2 = nn.Linear(hidden_dim, fusion_dim)
        
        # Branche 2: Pour les activations de la couche (ex: layer4)
        self.act_fc1 = nn.Linear(layer_activation_dim, hidden_dim)
        self.act_fc2 = nn.Linear(hidden_dim, fusion_dim)
        
        # Couche de fusion
        self.fusion_fc = nn.Linear(fusion_dim * 2, fusion_dim)
        
        # Couche de sortie pour prédire les spikes
        self.output_fc = nn.Linear(fusion_dim, spike_dim)
        
        # Fonctions d'activation et de régularisation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, obj_input, layer_activation):
        # Traitement de la branche des classes d'objets
        obj_x = self.embedding(obj_input)
        obj_x = self.dropout(self.relu(self.obj_fc1(obj_x)))
        obj_x = self.dropout(self.relu(self.obj_fc2(obj_x)))
        
        # Traitement de la branche des activations de couche
        act_x = self.dropout(self.relu(self.act_fc1(layer_activation)))
        act_x = self.dropout(self.relu(self.act_fc2(act_x)))
        
        # Fusion des deux branches (concaténation suivie d'une couche fully connected)
        combined = torch.cat([obj_x, act_x], dim=1)
        fusion = self.dropout(self.relu(self.fusion_fc(combined)))
        
        # Prédiction des spikes
        output = self.output_fc(fusion)
        
        return output
    
    def forward_objects_only(self, obj_input):
        """Méthode pour prédire en utilisant uniquement les classes d'objets"""
        obj_x = self.embedding(obj_input)
        obj_x = self.dropout(self.relu(self.obj_fc1(obj_x)))
        obj_x = self.dropout(self.relu(self.obj_fc2(obj_x)))
        
        # Pour la fusion, on remplace la partie activation par des zéros
        zero_act = torch.zeros_like(obj_x)
        combined = torch.cat([obj_x, zero_act], dim=1)
        fusion = self.dropout(self.relu(self.fusion_fc(combined)))
        
        output = self.output_fc(fusion)
        return output
    
    def forward_activations_only(self, layer_activation):
        """Méthode pour prédire en utilisant uniquement les activations de couche"""
        act_x = self.dropout(self.relu(self.act_fc1(layer_activation)))
        act_x = self.dropout(self.relu(self.act_fc2(act_x)))
        
        # Pour la fusion, on remplace la partie objet par des zéros
        zero_obj = torch.zeros_like(act_x)
        combined = torch.cat([zero_obj, act_x], dim=1)
        fusion = self.dropout(self.relu(self.fusion_fc(combined)))
        
        output = self.output_fc(fusion)
        return output

class CombinedSpikeDataset(Dataset):
    """
    Dataset qui contient à la fois les classes d'objets et les activations de couche.
    """
    def __init__(self, objects, layer_activations, spikes, device):
        # Gestion des objets qui peuvent être des chaînes de caractères
        if isinstance(objects[0], str):
            # Créer un mapping des chaînes vers des indices
            unique_objects = sorted(list(set(objects)))
            self.object_to_idx = {obj: idx for idx, obj in enumerate(unique_objects)}
            # Convertir les chaînes en indices
            object_indices = [self.object_to_idx[obj] for obj in objects]
            self.objects = torch.tensor(object_indices, dtype=torch.long, device=device)
        else:
            # Si les objets sont déjà numériques, les utiliser directement
            self.objects = torch.tensor(objects, dtype=torch.long, device=device)
            
        self.layer_activations = torch.tensor(layer_activations, dtype=torch.float32, device=device)
        self.spikes = torch.tensor(spikes, dtype=torch.float32, device=device)
        
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        return self.objects[idx], self.layer_activations[idx], self.spikes[idx]

def calculate_metrics(predictions, targets):
    """
    Calcule les métriques d'évaluation (variance expliquée et corrélation).
    """
    n_neurons = predictions.shape[1]
    explained_variance = np.zeros(n_neurons)
    
    for i in range(n_neurons):
        explained_variance[i] = explained_variance_score(targets[:, i], predictions[:, i])
    
    # Corrélation de Pearson pourrait également être calculée ici
    
    metrics = {
        'explained_variance': explained_variance,
        'explained_variance_mean': np.mean(explained_variance),
    }
    
    return metrics

def train_unified_model(model, train_dataset, val_dataset, num_epochs=20, batch_size=32, 
                        learning_rate=0.001, device='cuda'):
    """
    Fonction pour entraîner le modèle unifié.
    """
    # Vérifier si le modèle a besoin d'être mis à jour avec le bon nombre de classes
    if hasattr(train_dataset, 'object_to_idx'):
        num_classes = len(train_dataset.object_to_idx)
        if model.embedding.num_embeddings != num_classes:
            print(f"Mise à jour du nombre de classes d'objets dans le modèle: {num_classes}")
            # Recréer la couche d'embedding avec le bon nombre de classes
            old_embedding = model.embedding
            model.embedding = nn.Embedding(num_classes, old_embedding.embedding_dim).to(device)
            # Copier les poids existants pour les classes qui existaient déjà
            with torch.no_grad():
                min_classes = min(old_embedding.num_embeddings, num_classes)
                model.embedding.weight[:min_classes] = old_embedding.weight[:min_classes]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ev': [],
    }
    
    for epoch in range(num_epochs):
        # Mode entraînement
        model.train()
        train_loss = 0
        
        for obj_batch, act_batch, spikes_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(obj_batch, act_batch)
            loss = criterion(outputs, spikes_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Mode évaluation
        model.eval()
        with torch.no_grad():
            # Validation sur l'ensemble complet
            val_objects, val_activations, val_targets = (
                val_dataset.objects, 
                val_dataset.layer_activations, 
                val_dataset.spikes
            )
            
            # Prédictions avec le modèle complet
            val_outputs = model(val_objects, val_activations)
            val_loss = criterion(val_outputs, val_targets).item()
            
            # Prédictions avec chaque branche séparément
            obj_only_outputs = model.forward_objects_only(val_objects)
            act_only_outputs = model.forward_activations_only(val_activations)
            
            # Calcul des métriques
            unified_metrics = calculate_metrics(val_outputs.cpu().numpy(), val_targets.cpu().numpy())
            obj_only_metrics = calculate_metrics(obj_only_outputs.cpu().numpy(), val_targets.cpu().numpy())
            act_only_metrics = calculate_metrics(act_only_outputs.cpu().numpy(), val_targets.cpu().numpy())
            
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_ev'].append(unified_metrics['explained_variance_mean'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        print(f"Variance Expliquée - Unifié: {unified_metrics['explained_variance_mean']:.4f}, "
              f"Objets: {obj_only_metrics['explained_variance_mean']:.4f}, "
              f"Activations: {act_only_metrics['explained_variance_mean']:.4f}")
    
    return history

def extract_layer4_activations(model, images, device):
    """
    Extrait les activations de la couche 4 du ResNet.
    """
    model = model.to(device)
    model.eval()
    
    extractor = create_feature_extractor(model, return_nodes={'layer4': 'layer4'})
    extractor.eval()
    
    with torch.no_grad():
        activations = extractor(images)['layer4']
        # Aplatissez et réduisez la dimensionnalité si nécessaire
        activations = activations.mean(dim=[2, 3])  # Moyenne spatiale
        
    return activations.cpu().numpy()

def prepare_data_for_unified_model(stimulus_train, stimulus_val, object_train, object_val, spikes_train, spikes_val):
    """
    Prépare les données pour le modèle unifié en extrayant les activations de la couche 4.
    """
    # Conversion des stimuli en tenseurs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs_tr = torch.from_numpy(stimulus_train).to(device)
    imgs_val = torch.from_numpy(stimulus_val).to(device)
    
    # Chargement du modèle ResNet50 pré-entraîné
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()
    
    # Extraction des activations de la couche 4
    print("Extraction des activations de la couche 4 pour l'ensemble d'entraînement...")
    layer4_activations_tr = extract_layer4_activations(model, imgs_tr, device)
    print("Extraction des activations de la couche 4 pour l'ensemble de validation...")
    layer4_activations_val = extract_layer4_activations(model, imgs_val, device)
    
    # Création des datasets
    print("Création des datasets combinés...")
    train_dataset = CombinedSpikeDataset(object_train, layer4_activations_tr, spikes_train, device)
    
    # Pour assurer la cohérence des indices si les objets sont des chaînes
    if isinstance(object_train[0], str) and hasattr(train_dataset, 'object_to_idx'):
        # Convertir les objets de validation en utilisant le même mapping
        val_object_indices = []
        for obj in object_val:
            if obj in train_dataset.object_to_idx:
                val_object_indices.append(train_dataset.object_to_idx[obj])
            else:
                # Gérer les objets non vus dans l'ensemble d'entraînement
                print(f"Attention: Objet '{obj}' non vu dans l'ensemble d'entraînement.")
                # Assigner un indice par défaut (0 par exemple)
                val_object_indices.append(0)
        
        val_dataset = CombinedSpikeDataset(val_object_indices, layer4_activations_val, spikes_val, device)
    else:
        val_dataset = CombinedSpikeDataset(object_val, layer4_activations_val, spikes_val, device)
    
    return train_dataset, val_dataset 