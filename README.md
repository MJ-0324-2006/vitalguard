# VitalGuard: AI Early-Warning System for Patient Deterioration

**Team Orbit** — National Level Ideathon 5.0, CBIT Hyderabad (Smart Healthcare & Biomedical Innovation)

## Overview

VitalGuard is a lightweight early-warning system that detects patient deterioration trends using basic vital signs (heart rate, blood pressure, temperature, respiratory rate), designed for low-resource and rural hospitals with minimal infrastructure.

Built on real MIMIC-III clinical data, using multiple model architectures:

| Architecture | Role | Best Val AUROC |
|---|---|---|
| Transformer Detector | Multi-outcome classification | 0.7955 |
| Early Warning Model | Multi-task classification | 0.7751 |
| LSTM Autoencoder | Unsupervised anomaly scoring | (reconstruction loss based) |
| TCN Predictor | Multi-horizon vital sign forecasting | (forecast loss based) |

**Note:** These are our own results, trained and validated on MIMIC-III data as part of hackathon development — not peer-reviewed clinical results.

## Team
- Miriyala Jashmitha — AI/ML Lead
- Maddu Yamuna — UI/UX
- Maddela Aashritha — Data Handling
- Kothapalli Gowri Shankar — App Development
- Nukala Sai Chetan — Research & Presentation

## Disclaimer

This software is a hackathon prototype for research/educational purposes only. It is NOT approved for clinical use. See LICENSE for full terms.
