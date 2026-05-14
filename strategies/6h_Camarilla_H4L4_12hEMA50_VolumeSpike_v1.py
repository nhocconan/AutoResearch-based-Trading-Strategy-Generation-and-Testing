#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 breakout with 12h EMA50 trend filter and volume spike.
- Camarilla H4/L4 levels from 12h provide stronger institutional pivot points than R3/S3.
- 12h EMA50 trend filter ensures alignment with intermediate-term momentum.
- Volume spike (>2.0x 24-period average) confirms breakout validity.
- Discrete position sizing (0.25) minimizes fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) on 6h timeframe.
- Works in bull/bear via 12h trend filter and volatility-based volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla levels and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar (H, L, C)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla H4, L4 levels: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    camarilla_h4 = close_12h + (high_12h - low_12h) * 1.1 / 2
    camarilla_l4 = close_12h - (high_12h - low_12h) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (using previous completed 12h bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2.0x 24-period average volume (6h * 4 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H4 with volume spike and above 12h EMA50
            if close[i] > camarilla_h4_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L4 with volume spike and below 12h EMA50
            elif close[i] < camarilla_l4_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L4 OR below 12h EMA50
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla H4 OR above 12h EMA50
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0