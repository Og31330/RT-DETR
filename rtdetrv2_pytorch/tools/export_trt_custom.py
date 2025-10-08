import os
import argparse
import tensorrt as trt

# Taille de la VRAM du T4 : 16 GB. Allouons 8 GB (8 * 1024^3 bytes) pour le workspace.
T4_RECOMMENDED_WORKSPACE_GB = 8

def main(onnx_path, engine_path, max_batchsize, opt_batchsize, min_batchsize, use_fp16=True, verbose=False)->None:
    """ Convert ONNX model to TensorRT engine with T4-specific optimizations. """
    
    print("[DEBUG] 1. Démarrage de la fonction main.")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    print("[DEBUG] 2. Initialisation du Builder et Network.")
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    print("[DEBUG] 2b. Initialisation du OnnxParser.")
    parser = trt.OnnxParser(network, logger) 
    
    config = builder.create_builder_config() 
    
    # Correction de l'ancienne erreur de PreviewFeature (ligne commentée)
    # config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True) 

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    print(f"[INFO] 3. Loading ONNX file from {onnx_path}")
    
    print("[DEBUG] 4. Tentative de parser l'ONNX (opération critique)...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("\n[ERROR CRITIQUE] Failed to parse ONNX file. TensorRT reported errors:")
            for error in range(parser.num_errors):
                print(f"  [{error}] {parser.get_error(error)}") 
            raise RuntimeError("Failed to parse ONNX file")
    
    print("[DEBUG] 5. Parsing ONNX réussi. Préparation des configurations TRT...")

    # OPTIMISATION 1 : Augmenter la mémoire de travail (Workspace) - Correction d'API
    workspace_bytes = T4_RECOMMENDED_WORKSPACE_GB * (1 << 30) 
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    print(f"[INFO] Setting max_workspace_size (via set_memory_pool_limit) to {T4_RECOMMENDED_WORKSPACE_GB} GB for T4 optimization.")
    
    # OPTIMISATION 2 : Activer le FP16
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 optimization enabled (CRITICAL for T4 performance).")
        else:
            print("[WARNING] FP16 not supported on this platform. Proceeding with FP32.")

    # Définition du profil d'optimisation
    profile = builder.create_optimization_profile()
    
    # Entrée 1 : 'images' (Batch, Channel, Height, Width)
    profile.set_shape(
        "images", 
        min=(min_batchsize, 3, 640, 640), 
        opt=(opt_batchsize, 3, 640, 640), 
        max=(max_batchsize, 3, 640, 640)
    )
    
    # OPTIMISATION 3 : Correction de l'entrée 'orig_target_sizes' (Batch Dynamique)
    profile.set_shape(
        "orig_target_sizes", 
        min=(min_batchsize, 2), 
        opt=(opt_batchsize, 2), 
        max=(max_batchsize, 2)
    )
    
    config.add_optimization_profile(profile)

    print("[DEBUG] 6. Profil d'optimisation ajouté. Démarrage de la construction du moteur...") 
    print("[INFO] Building TensorRT engine (This may take several minutes)...")
    
    # CORRECTION D'API FINALE : Utiliser build_serialized_network()
    serialized_engine = builder.build_serialized_network(network, config) 

    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine. Check console logs for potential internal TRT errors.")

    print("[DEBUG] 7. Moteur construit et sérialisé.")
    print(f"[INFO] Saving engine to {engine_path}")
    
    # Écrire directement les bytes du moteur sérialisé
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print("[INFO] Engine export complete.")
    print("[DEBUG] 8. Fin de la fonction main.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine with T4 Optimizations")
    parser.add_argument("--onnx", "-i", type=str, required=True, help="Path to input ONNX model file")
    parser.add_argument("--saveEngine", "-o", type=str, default="rtdetrv2_t4_opt.engine", help="Path to output TensorRT engine file")
    
    # Paramètres de Batch par défaut
    parser.add_argument("--maxBatchSize", "-Mb", type=int, default=32, help="Maximum batch size for inference")
    parser.add_argument("--optBatchSize", "-Ob", type=int, default=16, help="Optimal batch size for inference (Recommended for T4 high-throughput)")
    parser.add_argument("--minBatchSize", "-mb", type=int, default=1, help="Minimum batch size for inference")
    
    parser.add_argument("--fp16", default=True, action="store_true", help="Enable FP16 precision mode (CRITICAL for T4)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.saveEngine)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # La casse de l'argument optBatchSize est corrigée.
    main(
        onnx_path=args.onnx,
        engine_path=args.saveEngine,
        max_batchsize=args.maxBatchSize,
        opt_batchsize=args.optBatchSize, 
        min_batchsize=args.minBatchSize,
        use_fp16=args.fp16,
        verbose=args.verbose
    )