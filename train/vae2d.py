"""Reuse the original VAE model definition from the release package in the training entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.vae2d import SimpleConditionalDenoisingVAE
