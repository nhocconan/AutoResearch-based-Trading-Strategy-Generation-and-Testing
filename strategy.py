# 12h_Combined_Pivot_Momentum_Signal_v1
# Hypothesis: Combines 1-day pivot point with 4-hour momentum (ROC) and volume confirmation.
# The idea is that price breaking above/below daily pivot with strong momentum and volume
# indicates institutional interest and trend continuation. Works in both bull and bear
# markets by capturing breakouts from key levels. Targets low trade frequency (15-30/year)
# to minimize fee drift while capturing strong moves.

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
    
    # Get 1-day data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Get 4-hour data for momentum calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4-hour ROC (Rate of Change) over 3 periods
    roc_4h = np.zeros_like(close_4h)
    roc_4h[3:] = (close_4h[3:] - close_4h[:-3]) / close_4h[:-3] * 100
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # Align all indicators to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    roc_4h_aligned = align_htf_to_ltf(prices, df_4h, roc_4h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_4h, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ROC and volume average
    start_idx = max(20, 3) + 1  # 20 for volume avg, 3 for ROC
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(roc_4h_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot = pivot_1d_aligned[i]
        roc = roc_4h_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Long: price above pivot with positive momentum and volume
            if close_val > pivot and roc > 0.5 and vol_conf:
                signals[i] = size
                position = 1
            # Short: price below pivot with negative momentum and volume
            elif close_val < pivot and roc < -0.5 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: price crosses below pivot or momentum turns negative
            if close_val < pivot or roc < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price crosses above pivot or momentum turns positive
            if close_val > pivot or roc > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Combined_Pivot_Momentum_Signal_v1"
timeframe = "12h"
leverage = 1.0