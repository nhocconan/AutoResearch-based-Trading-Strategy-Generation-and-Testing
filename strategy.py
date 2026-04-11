#!/usr/bin/env python3
# 4h_12h_camarilla_volume_reversal_v1
# Strategy: 4h reversal at 12h Camarilla pivot levels with volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from 12h timeframe act as strong support/resistance.
# Price reversals at these levels with above-average volume capture mean-reversion moves.
# Works in both bull and bear markets as it fades extremes rather than following trends.
# Low trade frequency (~25/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formula: based on previous day's range
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, etc.
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, etc.
    # We'll use S3 and R3 as primary levels (more significant)
    # S3 = C - (H-L)*1.1/4, R3 = C + (H-L)*1.1/4
    camarilla_s3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    camarilla_r3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Price near Camarilla levels (within 0.2% tolerance)
        near_s3 = abs(close[i] - camarilla_s3_aligned[i]) / close[i] < 0.002
        near_r3 = abs(close[i] - camarilla_r3_aligned[i]) / close[i] < 0.002
        
        # Reversal conditions
        # Long: Price near S3 support AND volume confirmation AND showing rejection (close > open)
        if near_s3 and vol_confirm and close[i] > prices['open'].iloc[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price near R3 resistance AND volume confirmation AND showing rejection (close < open)
        elif near_r3 and vol_confirm and close[i] < prices['open'].iloc[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price moves back toward middle (opposite Camarilla level) or opposite signal
        elif position == 1 and (close[i] > camarilla_r3_aligned[i] or close[i] < camarilla_s3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] < camarilla_s3_aligned[i] or close[i] > camarilla_r3_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals