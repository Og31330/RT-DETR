# Copyright 2023 lyuwenyu. All Rights Reserved.
# Copyright (c) 2025 Hitbee-dev. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
# NOTICE: This file has been heavily modified by [Hitbee-dev] from the original source.
# Modifications include restructuring for broader GPU architecture compatibility
# (including NVIDIA Blackwell), improved modularity, and enhanced testability.
# ==============================================================================

import time
import numpy as np
import torch
import tensorrt as trt
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont
import argparse
import torchvision.transforms as T
import os
import sys

# La taille d'entrée du modèle (RT-DETR utilise généralement 640x640)
INPUT_SIZE = 640

# ==============================================================================
# PARTIE 1: CLASSE TRTINFERENCE
# ==============================================================================

class TRTInference(object):
    """
    A high-level wrapper for TensorRT inference, designed for ease of use and flexibility.
    This class handles engine loading, context creation, and dynamic buffer allocation.
    """
    def __init__(self, engine_path, device='cuda:0', verbose=False):
        """
        Initializes the TRTInference instance.
        """
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.input_names, self.output_names = self._get_io_names()

        self.buffers_allocated = False
        self.gpu_buffers = OrderedDict()
        self.binding_addrs = OrderedDict()

        print(f"[TRTInference] Initialized successfully. Engine: '{engine_path}'.")

    def _load_engine(self, path):
        """Loads a TensorRT engine from a file."""
        with open(path, 'rb') as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from '{path}'.")
        return engine

    def _get_io_names(self):
        """Parses input and output tensor names from the engine."""
        input_names, output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        return input_names, output_names

    def _allocate_buffers(self, blob: dict):
        """
        Allocates GPU buffers for inputs and outputs based on the first inference request.
        """
        print("[TRTInference] First inference call detected. Allocating GPU buffers...")
        for name in self.input_names:
            tensor = blob[name]
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            self.context.set_input_shape(name, shape)
            self.gpu_buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
            self.binding_addrs[name] = self.gpu_buffers[name].data_ptr()
            print(f"  - Input '{name}': allocated buffer with shape {shape}.")

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.array(0, dtype=dtype)).dtype
            self.gpu_buffers[name] = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.binding_addrs[name] = self.gpu_buffers[name].data_ptr()
            print(f"  - Output '{name}': allocated buffer with shape {shape}.")

        self.buffers_allocated = True
        print("[TRTInference] GPU buffers allocated successfully.")

    def __call__(self, blob: dict):
        """
        Executes inference on the loaded TensorRT engine.
        """
        if not self.buffers_allocated:
            self._allocate_buffers(blob)
            
        for name in self.input_names:
            self.gpu_buffers[name].copy_(blob[name])

        self.context.execute_v2(bindings=list(self.binding_addrs.values()))
        
        return {name: self.gpu_buffers[name] for name in self.output_names}

# ==============================================================================
# PARTIE 2: POST-TRAITEMENT ET CORRECTION DES BBOX (NOUVEAU)
# ==============================================================================

def post_process_detections(boxes_gpu, w_orig, h_orig, input_size=INPUT_SIZE):
    """
    Applique la correction Y/X et la mise à l'échelle des Bounding Boxes.
    
    Args:
        boxes_gpu (torch.Tensor): Tensor GPU des boîtes de sortie, format probable [y1, x1, y2, x2]
                                  et mises à l'échelle dans l'espace [0, input_size].
        w_orig (int): Largeur originale de l'image.
        h_orig (int): Hauteur originale de l'image.
        input_size (int): Taille d'entrée du modèle (par défaut 640).

    Returns:
        torch.Tensor: Tensor GPU des boîtes au format [x1, y1, x2, y2] mises à l'échelle
                      aux dimensions originales de l'image.
    """
    if boxes_gpu.numel() == 0:
        return boxes_gpu # Rien à traiter
    
    # Étape 1: Créer une copie pour la modification (sur GPU)
    boxes_processed = boxes_gpu.clone()

    # Étape 2: Inversion Y/X (RT-DETR souvent [y1, x1, y2, x2] -> [x1, y1, x2, y2])
    # x1 <-> boxes_gpu[:, 1]
    # y1 <-> boxes_gpu[:, 0]
    # x2 <-> boxes_gpu[:, 3]
    # y2 <-> boxes_gpu[:, 2]
    
    # On crée un nouveau tensor pour les coordonnées réordonnées (sur GPU)
    boxes_reordered = torch.empty_like(boxes_gpu)
    boxes_reordered[:, 0] = boxes_processed[:, 1] # x1
    boxes_reordered[:, 1] = boxes_processed[:, 0] # y1
    boxes_reordered[:, 2] = boxes_processed[:, 3] # x2
    boxes_reordered[:, 3] = boxes_processed[:, 2] # y2
    
    # Étape 3: Mise à l'échelle des coordonnées [0, INPUT_SIZE] -> [0, W_orig/H_orig]
    scale_w = float(w_orig) / input_size
    scale_h = float(h_orig) / input_size
    
    # Mise à l'échelle des coordonnées X (colonnes 0 et 2)
    boxes_reordered[:, [0, 2]] *= scale_w 
    # Mise à l'échelle des coordonnées Y (colonnes 1 et 3)
    boxes_reordered[:, [1, 3]] *= scale_h 
    
    print(f"  - BBox Scales (W, H): ({scale_w:.4f}, {scale_h:.4f})")
    
    # Debug trace (afficher la première boîte avant/après correction)
    try:
        box_raw = boxes_gpu[0].cpu().numpy()
        box_scaled = boxes_reordered[0].cpu().numpy()
        print(f"  - DEBUG BBox brute (y1, x1, y2, x2, normalisée {input_size}): {box_raw.round(2)}")
        print(f"  - DEBUG BBox finale (x1, y1, x2, y2, à l'échelle {w_orig}x{h_orig}): {box_scaled.round(2)}")
    except Exception:
        pass # Ignorer si aucune boîte trouvée

    return boxes_reordered


# ==============================================================================
# PARTIE 3: VISUALISATION ET MAIN
# ==============================================================================

# --- Visualization Utility Function ---
COCO_CLASSES = [
    'background', 'ball', 'shot']

def visualize_detections(image_pil, boxes, scores, labels, class_names=COCO_CLASSES, threshold=0.5):
    """
    Draws bounding boxes on a PIL image. Assumes boxes are already scaled to image_pil size.
    """
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Ensure tensors are on CPU and converted to NumPy for processing
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    
    count = 0
    for i in range(len(scores)):
        score = scores[i]
        if score < threshold:
            continue
        
        count += 1
        box = boxes[i]
        label_idx = int(labels[i])
        
        # xmin, ymin, xmax, ymax sont déjà à l'échelle de l'image
        xmin, ymin, xmax, ymax = box
        class_name = class_names[label_idx] if label_idx < len(class_names) else f'CLS-{label_idx}'
        color = 'red' # Keep it simple or use a color map
        
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=3)
        
        text = f"{class_name}: {score:.2f}"
        
        try:
            # Essayer de charger une police TrueType pour un meilleur affichage
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Calculer la position du texte et du fond
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        # S'assurer que le texte ne sort pas de l'image en haut
        text_y_pos = max(ymin - 20, 0) 
        
        # Dessin du fond du texte
        draw.rectangle((xmin, text_y_pos, text_bbox[2], text_y_pos + 20), fill=color)
        # Dessin du texte
        draw.text((xmin, text_y_pos), text, fill="white", font=font)
        
    print(f"  - Found {count} objects above threshold {threshold}.")
    return img_draw

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Test script for the TRTInference wrapper.")
    parser.add_argument('--engine', type=str, required=True, help="Path to the TensorRT engine file.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--output', type=str, default='output.jpg', help="Path to save the output image with detections.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run inference on.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold for displaying detections.")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Erreur: CUDA n'est pas disponible. Ce script nécessite un GPU.")
        sys.exit(1)
    
    print("--- TRTInference Wrapper Test ---")
    
    print("\n1. Initializing TRTInference...")
    try:
        trt_model = TRTInference(args.engine, device=args.device)
    except Exception as e:
        print(f"Erreur fatale lors du chargement du moteur TRT: {e}")
        sys.exit(1)
    
    print("\n2. Preprocessing input image...")
    try:
        image_pil = Image.open(args.image).convert('RGB')
    except FileNotFoundError:
        print(f"Erreur: Image non trouvée à l'emplacement: {args.image}")
        sys.exit(1)
        
    w, h = image_pil.size
    
    # --- PRÉ-TRAITEMENT ---
    transforms = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
    ])
    
    image_tensor = transforms(image_pil).unsqueeze(0).to(args.device)
    orig_size_tensor = torch.tensor([[h, w]], dtype=torch.int64, device=args.device) # H, W car c'est le format attendu par RT-DETR
    
    blob = {
        'images': image_tensor,
        'orig_target_sizes': orig_size_tensor
    }
    print(f"  - Original image size: {w}x{h}")
    print(f"  - Input tensor shape: {image_tensor.shape}")

    print("\n3. Running inference...")
    start_time = time.time()
    output_gpu = trt_model(blob)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"\n4. Inference complete in { (end_time - start_time) * 1000:.2f} ms.")
    
    print("\n5. Post-processing and saving output image...")
    
    # 5a. Extraction des résultats
    output_labels = output_gpu['labels'][0]
    output_boxes_raw = output_gpu['boxes'][0]
    output_scores = output_gpu['scores'][0]
    
    # 5b. Filtrage par seuil
    valid_idx = output_scores > args.threshold
    labels = output_labels[valid_idx]
    boxes_raw = output_boxes_raw[valid_idx]
    scores = output_scores[valid_idx]
    
    # 5c. CORRECTION CRUCIALE DES BBOX
    if boxes_raw.numel() > 0:
        output_boxes_corrected = post_process_detections(boxes_raw, w, h, INPUT_SIZE)
    else:
        print("  - Aucune détection à traiter après le filtrage par seuil.")
        output_boxes_corrected = boxes_raw # Tensor vide
        
    # 5d. Utilisation des boîtes CORRIGÉES pour la visualisation
    result_image = visualize_detections(
        image_pil, 
        output_boxes_corrected, 
        scores, 
        labels, 
        threshold=args.threshold
    )
    
    result_image.save(args.output)
    print(f"  - Output image with detections saved to: {os.path.abspath(args.output)}")

    print("\n--- Test finished successfully ---")