#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Pullback_1dEMA34_Trend_VolumeConfirm
Hypothesis: Trade pullbacks to Camarilla R1/S1 levels in direction of 1d EMA34 trend with volume confirmation (>1.5x 20-bar MA). This reduces false breakouts by requiring alignment with higher timeframe trend and institutional volume. Discrete sizing 0.25 limits fee drag. Target 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    camarilla_R1 = close_1d + camarilla_range
    camarilla_S1 = close_1d - camarilla_range
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA34 to 4h (completed 1d bar only)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (1d), EMA34 (1d), volume MA (20)
    start_idx = max(34, 20)  # 34 for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: pullback to Camarilla R1 support in uptrend + volume confirmation
            long_setup = (close[i] <= camarilla_R1_aligned[i] * 1.005) and \
                         (close[i] >= camarilla_R1_aligned[i] * 0.995) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_confirm[i]
            # Short: pullback to Camarilla S1 resistance in downtrend + volume confirmation
            short_setup = (close[i] <= camarilla_S1_aligned[i] * 1.005) and \
                          (close[i] >= camarilla_S1_aligned[i] * 0.995) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below Camarilla S1 OR below 1d EMA34
            if (close[i] < camarilla_S1_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Camarilla R1 OR above 1d EMA34
            if (close[i] > camarilla_R1_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Pullback_1dEMA34_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0