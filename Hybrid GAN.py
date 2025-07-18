#!/usr/bin/env python
# coding: utf-8

# In[1]:

import datetime
# Get current date and time
current_time = datetime.datetime.now()
# Print date, time, and day in the desired format
print("This notebooks code execution is starting at:")
print("Time :", current_time.strftime("%I:%M:%S %p"))  # 12-hour format with AM/PM
print("Date :", current_time.strftime("%d-%m-%Y"))
print("Day  :", current_time.strftime("%A"))

# In[2]:

import os
import shutil
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Embedding, Concatenate, LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import subprocess
from google.colab import drive

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# =============================
# GPU Configuration
# =============================
def configure_gpu():
    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)
    
    # Set mixed precision policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled.")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")

        print(f"Available GPUs: {len(physical_devices)}")
        try:
            gpu_names = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']
            ).decode('utf-8').strip().split('\n')
            for idx, name in enumerate(gpu_names):
                print(f"GPU {idx}: {name}")
        except Exception as e:
            print(f"Error getting GPU names: {e}")
    else:
        print("No GPUs detected")

configure_gpu()
print("\n")   

# =============================
# Configuration & Hyperparameters
# =============================
DATASET_PATH = "/content/drive/MyDrive/Project BTD (GAN)/Important Brain MRI Dataset/Training"
OUTPUT_PATH  = "/content/drive/MyDrive/Results Nvidia RTX 3080 TI/Hybrid GAN"

# Clear the output folder at each run
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Image dimensions and channels (grayscale)
IMG_HEIGHT = 512
IMG_WIDTH  = 512
CHANNELS   = 1

# GAN Hyperparameters for WGAN-GP
LATENT_DIM      = 100        # Dimension of random noise vector
NUM_CLASSES     = 4          # glioma, meningioma, no tumor, pituitary
BATCH_SIZE      = 128       # Increased from 4 to utilize GPU better
EPOCHS          = 500        
N_DISCRIMINATOR = 1          
GP_WEIGHT       = 1          

# Learning rates
GENERATOR_LR    = 1e-4
DISCRIMINATOR_LR = 1e-4

# =============================
# Data Loading & Preprocessing (Optimized)
# =============================
def load_dataset(dataset_path, batch_size):
    raw_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True
    )
    normalization_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
    normalized_dataset = raw_dataset.map(lambda images, labels: (normalization_layer(images), labels))
    return raw_dataset, normalized_dataset.cache().prefetch(tf.data.AUTOTUNE)

raw_dataset, train_dataset = load_dataset(DATASET_PATH, BATCH_SIZE)

print(f"\nDataset Path: {DATASET_PATH}")
print("\nClass names and their corresponding values:")
for index, class_name in enumerate(raw_dataset.class_names):
    print(f"Label Value: {index}, Label Name: {class_name}")

def count_images_per_folder(dataset_path):
    folder_details = []
    for root, dirs, files in os.walk(dataset_path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            image_count = len([f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))])
            folder_details.append({"folder_path": folder_path, "image_count": image_count})
    return folder_details

folder_details = count_images_per_folder(DATASET_PATH)
print("\nFolders and Image Counts:")
for folder in folder_details:
    print(f"Folder Path: {folder['folder_path']}, Total Images: {folder['image_count']}")
    
# =============================
# Build Models (With Mixed Precision Adjustments)
# =============================
def build_generator():
    noise_input = Input(shape=(LATENT_DIM,), name="gen_noise")
    label_input = Input(shape=(1,), dtype='int32', name="gen_label")
    label_embedding = Embedding(NUM_CLASSES, LATENT_DIM, input_length=1)(label_input)
    label_embedding = Flatten()(label_embedding)
    combined_input = Concatenate()([noise_input, label_embedding])
    
    x = Dense(8 * 8 * 512, use_bias=False)(combined_input)
    x = Reshape((8, 8, 512))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Intermediate layers will use mixed precision automatically
    x = Conv2DTranspose(256, 4, 2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2DTranspose(128, 4, 2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2DTranspose(64, 4, 2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2DTranspose(32, 4, 2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2DTranspose(16, 4, 2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2DTranspose(8, 4, 2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Final layer with explicit float32 dtype
    output = Conv2DTranspose(CHANNELS, 4, 1, padding='same',
                             activation='tanh', use_bias=False, dtype='float32')(x)
    return Model([noise_input, label_input], output, name="Generator")

def build_discriminator():
    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="discriminator_image")
    label_input = Input(shape=(1,), dtype='int32', name="discriminator_label")
    label_embedding = Embedding(NUM_CLASSES, IMG_HEIGHT * IMG_WIDTH)(label_input)
    label_embedding = Flatten()(label_embedding)
    label_embedding = Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(label_embedding)
    x = Concatenate(axis=-1)([image_input, label_embedding])
    
    x = Conv2D(64, 4, 2, padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, 4, 2, padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(256, 4, 2, padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(512, 4, 2, padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(1024, 4, 2, padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(2048, 4, 2, padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    output = Dense(1, dtype='float32')(x)  # Final layer in float32
    return Model([image_input, label_input], output, name="Discriminator")

# =============================
# HybridGAN Model (With Graph Execution)
# =============================
class HybridGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Trackers for loss components
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.gp_tracker = tf.keras.metrics.Mean(name="gradient_penalty")

    
    def call(self, inputs):
        """Defines the forward pass for inference"""
        noise, labels = inputs
        return self.generator([noise, labels])

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.gp_tracker]

    def compile(self, g_optimizer, d_optimizer, **kwargs):
        super().compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    
    def gradient_penalty(self, real_images, fake_images, labels):
        # Cast all components to float32 explicitly
        real_images = tf.cast(real_images, tf.float32)
        fake_images = tf.cast(fake_images, tf.float32)

        epsilon = tf.random.uniform(
            shape=[tf.shape(real_images)[0]], 
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32  # Force float32 calculation
        )
        epsilon = tf.reshape(epsilon, [tf.shape(epsilon)[0], 1, 1, 1])

        # Mixed precision-safe interpolation
        interpolated = (epsilon * real_images + 
                       (1 - epsilon) * fake_images)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # Cast interpolated to mixed precision policy dtype
            pred = self.discriminator([
                tf.cast(interpolated, tf.float16),  # Match policy dtype
                labels
            ], training=True)

        grads = tape.gradient(pred, [interpolated])[0]
        grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]) + 1e-12)
        gp = tf.reduce_mean((grads_norm - 1.0) ** 2)
        return gp
    
    
    def train_step(self, data):
        real_images, labels = data
        
        # Train discriminator
        d_loss_total = 0
        for _ in range(N_DISCRIMINATOR):
            noise = tf.random.normal([tf.shape(real_images)[0], self.latent_dim])
            
            with tf.GradientTape() as tape:
                fake_images = self.generator([noise, labels], training=True)
                real_output = self.discriminator([real_images, labels], training=True)
                fake_output = self.discriminator([fake_images, labels], training=True)
                
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gp = self.gradient_penalty(real_images, fake_images, labels)
                d_loss_total = d_loss + GP_WEIGHT * gp
                
            d_gradients = tape.gradient(d_loss_total, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        noise = tf.random.normal([tf.shape(real_images)[0], self.latent_dim])
        with tf.GradientTape() as tape:
            generated_images = self.generator([noise, labels], training=True)
            gen_output = self.discriminator([generated_images, labels], training=True)
            g_loss = -tf.reduce_mean(gen_output)
            
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Update metrics
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss_total)
        self.gp_tracker.update_state(gp)
        
        return {m.name: m.result() for m in self.metrics}

# =============================
# Model Initialization
# =============================
generator = build_generator()
discriminator = build_discriminator()

Hybrid_GAN = HybridGAN(
    generator=generator,
    discriminator=discriminator,
    latent_dim=LATENT_DIM,
    num_classes=NUM_CLASSES
)

# Build with generator's input signature
Hybrid_GAN.build([
    (None, LATENT_DIM),  # Noise input shape
    (None, 1)            # Label input shape
])

Hybrid_GAN.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=GENERATOR_LR, beta_1=0.5, beta_2=0.9),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=DISCRIMINATOR_LR, beta_1=0.5, beta_2=0.9)
)

# =============================
# Display Model Summaries
# =============================
print("\n" + "="*60)
print("HybridGAN Summary")
print("="*60)
Hybrid_GAN.summary()

print("\n" + "="*60)
print("Generator Architecture")
print("="*60)
generator.summary()

print("\n" + "="*60)
print("Discriminator Architecture")
print("="*60)
discriminator.summary()

# =============================
# Training Loop
# =============================
print("\nTraining Start...")
epoch_loss_data = []

for epoch in range(EPOCHS):
    start = time.time()
    epoch_batches = 0
    
    for batch in train_dataset:
        metrics = Hybrid_GAN.train_step(batch)
        epoch_batches += 1

    epoch_time = time.time() - start
    metrics = {k: v.numpy() for k, v in metrics.items()}
    
    epoch_loss_data.append((
        epoch, 
        metrics['d_loss'], 
        metrics['g_loss'], 
        epoch_time, 
        epoch_batches
    ))

    print(f"Epoch: {epoch} | Batches: {epoch_batches} | "
          f"D Loss: {metrics['d_loss']:.4f} | G Loss: {metrics['g_loss']:.4f} | "
          f"GP: {metrics['gradient_penalty']:.4f} | Time: {epoch_time:.2f}s")

    # Sample generation
    if epoch % 5 == 0:
        sample_noise = tf.random.normal([NUM_CLASSES, LATENT_DIM])
        sample_labels = tf.range(NUM_CLASSES)[:, None]
        sample_images = generator([sample_noise, sample_labels], training=False)
        sample_images = ((sample_images + 1) * 127.5).numpy().astype(np.uint8)
        
        sample_dir = os.path.join(OUTPUT_PATH, "Samples")
        os.makedirs(sample_dir, exist_ok=True)
        for i, img in enumerate(sample_images):
            Image.fromarray(img.squeeze(), 'L').save(
                os.path.join(sample_dir, f"Epoch_{epoch}_Class_{i}.jpg")
            )

print("\nTraining Completed...")
# =============================
# Post-Training Analysis
# =============================
# Plot training curves
plt.figure(figsize=(12, 6))
plt.plot([x[1] for x in epoch_loss_data], label='Discriminator Loss')
plt.plot([x[2] for x in epoch_loss_data], label='Generator Loss')
plt.title("Training Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, "training_curves.png"))
plt.close()

# In[3]:

# =============================
# Generate and Save Augmented Images
# =============================
def count_images_in_class(class_path):
    """
    Returns the total number of images (jpg, jpeg, png) in the given folder.
    """
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(class_path, ext)))
    return len(files)

def generate_augmented_images():
    """
    For each class folder in DATASET_PATH, generate the same number of new images
    as originally present. Save these generated grayscale images in OUTPUT_PATH/<class_name>/.
    """
    class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    for class_name in class_names:
        class_dir = os.path.join(DATASET_PATH, class_name)
        num_images = count_images_in_class(class_dir)
        print(f"Generating images for class '{class_name}' (Total: {num_images})")
        out_class_dir = os.path.join(OUTPUT_PATH, "Training", class_name)
        # Clear the output folder at each run
        if os.path.exists(out_class_dir):
            shutil.rmtree(out_class_dir)
        os.makedirs(out_class_dir, exist_ok=True)
        batch_gen = 16
        num_generated = 0
        while num_generated < num_images:
            curr_batch = min(batch_gen, num_images - num_generated)
            noise = tf.random.normal([curr_batch, LATENT_DIM])
            labels = tf.constant([[class_names.index(class_name)]] * curr_batch, dtype=tf.int32)
            generated = generator([noise, labels], training=False)
            generated = ((generated + 1) * 127.5)
            generated = tf.cast(generated, tf.uint8).numpy()
            for i in range(curr_batch):
                im = Image.fromarray(generated[i].squeeze(), mode="L")
                save_path = os.path.join(out_class_dir, f"{num_generated + i + 1}.jpg")
                im.save(save_path)
            num_generated += curr_batch

generate_augmented_images()
print("Augmented image generation complete!")

# In[4]:

# =============================
# XLA/CUDA Configuration Fix
# =============================
import os
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={os.path.dirname(os.path.dirname(os.path.dirname(tf.__file__)))}'
tf.config.optimizer.set_jit(False)  # Disable XLA compilation

# Add this BEFORE creating GANAnalyser instance
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Memory growth enabled successfully.")
    except:
        print("Error in memory growth enbaling.")
        pass

# In[5]:

import os
import gc
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from tensorflow.keras.applications import InceptionV3

from tensorflow.keras import backend as K
K.clear_session()
gc.collect()

class GANAnalyser:
    def __init__(self, real_dir, gen_dir, input_shape=(512, 512, 1), batch_size=32):
        self.real_dir = real_dir
        self.gen_dir = gen_dir
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.inception = InceptionV3(include_top=False, pooling='avg', input_shape=(512, 512, 3))
        self.valid_classes = self._validate_class_folders()

    def _validate_class_folders(self):
        """Validate and get matching class folders with equal image counts"""
        valid_classes = []
        
        real_classes = {d.lower(): (d, len(glob(os.path.join(self.real_dir, d, "*")))) 
                       for d in os.listdir(self.real_dir) if os.path.isdir(os.path.join(self.real_dir, d))}
        
        gen_classes = {d.lower(): (d, len(glob(os.path.join(self.gen_dir, d, "*")))) 
                      for d in os.listdir(self.gen_dir) if os.path.isdir(os.path.join(self.gen_dir, d))}

        for cls_lower in real_classes:
            if cls_lower in gen_classes:
                real_name, real_count = real_classes[cls_lower]
                gen_name, gen_count = gen_classes[cls_lower]
                if real_count == gen_count and real_count > 0:
                    valid_classes.append((real_name, gen_name, real_count))
                else:
                    print(f"Skipping {real_name}: {real_count} real vs {gen_count} generated")
        return sorted(valid_classes, key=lambda x: x[0])

    def _load_batch(self, real_paths, gen_paths):
        """Load batch of image pairs with memory optimization"""
        real_batch = []
        gen_batch = []
        
        for rp, gp in zip(real_paths, gen_paths):
            real_img = Image.open(rp).convert('L').resize(self.input_shape[:2])
            gen_img = Image.open(gp).convert('L').resize(self.input_shape[:2])
            
            real_batch.append(np.array(real_img, dtype=np.float32) / 255.0)
            gen_batch.append(np.array(gen_img, dtype=np.float32) / 255.0)
            
        return np.array(real_batch), np.array(gen_batch)

    def _process_dataset(self, process_batch_fn, paths_list=None):
        """Memory-efficient dataset processor"""
        metrics = {
            'mse_sum': 0.0, 'mae_sum': 0.0, 'ssim_sum': 0.0, 'psnr_sum': 0.0,
            'real_cnr_sum': 0.0, 'gen_cnr_sum': 0.0,
            'real_features': [], 'gen_features': [], 
            'real_samples': [], 'gen_samples': [],
            'count': 0
        }
        
        process_generator = self.valid_classes if paths_list is None else paths_list
        
        for real_name, gen_name, total_count in process_generator:
            real_paths = sorted(glob(os.path.join(self.real_dir, real_name, "*")))
            gen_paths = sorted(glob(os.path.join(self.gen_dir, gen_name, "*")))
            
            for i in tqdm(range(0, len(real_paths), self.batch_size), 
                         desc=f"Processing of Class: {real_name}" if paths_list else f"Overall Processing of Class: {real_name}"):
                batch_real_paths = real_paths[i:i+self.batch_size]
                batch_gen_paths = gen_paths[i:i+self.batch_size]
                
                real_batch, gen_batch = self._load_batch(batch_real_paths, batch_gen_paths)
                metrics = process_batch_fn(real_batch, gen_batch, metrics)
                
                if len(metrics['real_samples']) < 100:
                    metrics['real_samples'].extend(real_batch)
                    metrics['gen_samples'].extend(gen_batch)
                
                del real_batch, gen_batch
                gc.collect()
                
        return metrics

    def calculate_metrics(self):
        """Calculate metrics for complete dataset and individual classes"""
        # 1. Calculate overall metrics using all classes
        print("\n{:=^80}".format(" Calculating Overall Dataset Metrics "))
        overall_metrics = self._calculate_metrics()
        
        # 2. Calculate per-class metrics
        print("\n{:=^80}".format(" Calculating Per-Class Metrics "))
        class_metrics = {}
        
        for class_info in self.valid_classes:
            real_name = class_info[0]
            print(f"\n{'>'*10} Processing Class: {real_name} {'<'*10}")
            class_result = self._calculate_metrics(paths_list=[class_info])
            class_metrics[real_name] = class_result
        
        return {'overall': overall_metrics, 'classes': class_metrics}

    def _calculate_metrics(self, paths_list=None):
        """Core metric calculation logic"""
        def process_batch(real_batch, gen_batch, metrics):
            # Basic metrics
            metrics['mse_sum'] += np.sum(mean_squared_error(real_batch, gen_batch))
            metrics['mae_sum'] += np.sum(np.mean(np.abs(real_batch - gen_batch), axis=(1,2)))
            metrics['ssim_sum'] += np.sum([structural_similarity(r, g, data_range=1.0) 
                                         for r, g in zip(real_batch, gen_batch)])
            metrics['psnr_sum'] += np.sum([peak_signal_noise_ratio(r, g) 
                                         for r, g in zip(real_batch, gen_batch)])
            
            # CNR metrics
            metrics['real_cnr_sum'] += np.sum([np.mean(img)/np.std(img) for img in real_batch])
            metrics['gen_cnr_sum'] += np.sum([np.mean(img)/np.std(img) for img in gen_batch])
            
            # Feature extraction for FID/KID
            real_features = self.inception.predict(np.repeat(real_batch[..., np.newaxis], 3, axis=-1))
            gen_features = self.inception.predict(np.repeat(gen_batch[..., np.newaxis], 3, axis=-1))
            metrics['real_features'].append(real_features)
            metrics['gen_features'].append(gen_features)
            
            metrics['count'] += len(real_batch)
            return metrics
        
        metrics = self._process_dataset(process_batch, paths_list)
        total = metrics['count']
        
        # Calculate basic metrics
        results = {
            'MSE': metrics['mse_sum'] / total,
            'MAE': metrics['mae_sum'] / total,
            'SSIM': metrics['ssim_sum'] / total,
            'PSNR': metrics['psnr_sum'] / total,
            'RMSE': np.sqrt(metrics['mse_sum'] / total),
            'CNR (Real)': metrics['real_cnr_sum'] / total,
            'CNR (Generated)': metrics['gen_cnr_sum'] / total,
        }
        
        # Combine features for deep metrics
        if metrics['real_features']:
            real_features = np.concatenate(metrics['real_features'])
            gen_features = np.concatenate(metrics['gen_features'])
            
            # Fréchet Inception Distance
            mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
            mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
            results['FID'] = np.sum((mu_real - mu_gen)**2) + np.trace(sigma_real + sigma_gen - 2*sqrtm(sigma_real@sigma_gen).real)
            
            # Kernel Inception Distance
            m, n = len(real_features), len(gen_features)
            results['KID'] = (polynomial_kernel(real_features).sum()/(m*(m-1)) + 
                             polynomial_kernel(gen_features).sum()/(n*(n-1)) - 
                             2*polynomial_kernel(real_features, gen_features).sum()/(m*n))
        
        # Distance Correlation (using first 100 samples)
        results['DC'] = self._distance_correlation(
            np.array(metrics['real_samples'][:100]), 
            np.array(metrics['gen_samples'][:100])
        ) if metrics['real_samples'] else 0.0
        
        return results

    def _distance_correlation(self, X, Y):
        """Calculate distance correlation between sample pairs"""
        def _centered_distances(mat):
            d = squareform(pdist(mat.reshape(len(mat), -1)))  # Flatten images
            return d - d.mean(axis=1, keepdims=True) - d.mean(axis=0) + d.mean()
        
        A = _centered_distances(X)
        B = _centered_distances(Y)
        
        dcov = np.sqrt(np.mean(A * B))
        dvar_x = np.sqrt(np.mean(A**2))
        dvar_y = np.sqrt(np.mean(B**2))
        
        return dcov / np.sqrt(dvar_x * dvar_y) if (dvar_x * dvar_y) > 0 else 0.0

def print_metrics(results):
    """Print formatted metrics for both overall and per-class results"""
    ideal_values = {
        'MSE': 'Lower (0 ideal)',
        'MAE': 'Lower (0 ideal)',
        'SSIM': 'Higher (1 ideal)',
        'PSNR': 'Higher (>30 dB)',
        'RMSE': 'Lower (0 ideal)',
        'CNR (Real)': 'Higher',
        'CNR (Generated)': 'Higher',
        'FID': 'Lower (<50 good)',
        'KID': 'Lower (0 ideal)',
        'DC': 'Higher (1 ideal)'
    }
    
    # Print overall metrics
    print("\n{:=^80}".format(" Overall Dataset Metrics "))
    print("{:<25} {:<20} {:<35}".format("Metric", "Value", "Ideal Value"))
    print("-"*80)
    for metric, value in results['overall'].items():
        print(f"{metric:<25} {value:<20.4f} {ideal_values.get(metric, ''):<35}")
    
    # Print per-class metrics
    for class_name, metrics in results['classes'].items():
        print("\n{:=^80}".format(f" {class_name} Class Metrics "))
        print("{:<25} {:<20} {:<35}".format("Metric", "Value", "Ideal Value"))
        print("-"*80)
        for metric, value in metrics.items():
            print(f"{metric:<25} {value:<20.4f} {ideal_values.get(metric, ''):<35}")
    
    print("="*80 + "\n")

# Usage Example
if __name__ == "__main__":
    DATASET_PATH = "/content/drive/MyDrive/Project BTD(GAN)/Important Brain MRI Dataset/Training"
    OUTPUT_PATH = "/content/drive/MyDrive/Results Nvidia RTX 3080 TI/Hybrid GAN/Training"
    
    try:
        analyser = GANAnalyser(DATASET_PATH, OUTPUT_PATH, batch_size=16)
        results = analyser.calculate_metrics()
        print_metrics(results)
    except Exception as e:
        print(f"Error: {str(e)}")

# In[6]:

import datetime
# Get current date and time
current_time = datetime.datetime.now()
# Print date, time, and day in the desired format
print("This notebooks code execution is ending at:")
print("Time :", current_time.strftime("%I:%M:%S %p"))  # 12-hour format with AM/PM
print("Date :", current_time.strftime("%d-%m-%Y"))
print("Day  :", current_time.strftime("%A"))
