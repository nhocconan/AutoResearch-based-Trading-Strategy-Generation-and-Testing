#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform with 1d trend filter and volume confirmation
# Long when Fisher crosses above -1.5 AND price > 1d EMA50 AND volume > 1.3x 6h average volume
# Short when Fisher crosses below +1.5 AND price < 1d EMA50 AND volume > 1.3x 6h average volume
# Fisher Transform identifies turning points in cyclical price action
# EMA50 filter ensures alignment with intermediate trend to avoid counter-trend trades
# Volume confirmation adds conviction to reversals
# Target: 80-160 total trades over 4 years (20-40/year) with controlled risk

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Ehlers Fisher Transform (9-period) ===
    # Normalize price to 0-1 range over lookback period
    hlc = (high + low + close) / 3.0
    max_h = pd.Series(hlc).rolling(window=9, min_periods=9).max().values
    min_l = pd.Series(hlc).rolling(window=9, min_periods=9).min().values
    range_hlc = max_h - min_l
    # Avoid division by zero
    range_hlc = np.where(range_hlc == 0, 1e-10, range_hlc)
    value1 = 0.33 * 2 * ((hlc - min_l) / range_hlc - 0.5) + 0.67 * np.roll(0.33 * 2 * ((hlc - min_l) / range_hlc - 0.5) + 0.67 * np.zeros_like(hlc), 1)
    value1[0] = 0
    # Apply smoothing
    value2 = pd.Series(value1).ewm(alpha=0.5, adjust=False).mean().values
    # Fisher Transform
    value2 = np.clip(value2, -0.999, 0.999)
    fisher = 0.5 * np.log((1 + value2) / (1 - value2))
    # Signal line
    fisher_signal = pd.Series(fisher).ewm(span=3, adjust=False).mean().values
    
    # === Volume Confirmation (6h average) ===
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values  # 6 periods of 6h = 36h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(fisher[i]) or
            np.isnan(fisher_signal[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        fish = fisher[i]
        fish_sig = fisher_signal[i]
        vol_confirm = volume[i] > vol_ma[i] * 1.3  # 1.3x average volume
        
        # Fisher crossover signals
        fish_cross_up = fish > fish_sig and np.roll(fish, 1)[i] <= np.roll(fisher_signal, 1)[i]
        fish_cross_down = fish < fish_sig and np.roll(fish, 1)[i] >= np.roll(fisher_signal, 1)[i]
        
        # Handle roll boundary
        if i == 0:
            fish_cross_up = False
            fish_cross_down = False
        else:
            fish_cross_up = fish > fish_sig and fisher[i-1] <= fisher_signal[i-1]
            fish_cross_down = fish < fish_sig and fisher[i-1] >= fisher_signal[i-1]
        
        # === ENTRY LOGIC ===
        if position == 0:
            # Long when: Fisher crosses above signal AND price > EMA50 AND volume confirmation
            if fish_cross_up and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Fisher crosses below signal AND price < EMA50 AND volume confirmation
            elif fish_cross_down and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # === EXIT LOGIC: Reverse on opposite signal ===
        elif position == 1 and fish_cross_down:
            signals[i] = -0.25
            position = -1
        elif position == -1 and fish_cross_up:
            signals[i] = 0.25
            position = 1
        
        # Hold current position
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_FisherTransform_1dEMA50_Volume1.3x"
timeframe = "6h"
leverage = 1.0