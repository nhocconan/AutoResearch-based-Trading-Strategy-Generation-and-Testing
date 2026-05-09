#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot direction and daily volume confirmation.
# Uses 1w pivot points (calculated from prior week) for trend filter and 1d volume spike for entry timing.
# Designed for low trade frequency (12-37/year) to avoid fee drag in 6h timeframe.
# Works in both bull/bear markets by requiring alignment with weekly pivot trend and volume confirmation.
name = "6h_WeeklyPivot_DailyVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # R1 = 2*P - L, S1 = 2*P - H, R2 = P + (H-L), S2 = P - (H-L)
    # where P = (H+L+C)/3
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    wp = (wh + wl + wc) / 3.0
    wr1 = 2 * wp - wl
    ws1 = 2 * wp - wh
    wr2 = wp + (wh - wl)
    ws2 = wp - (wh - wl)
    
    # Align weekly pivots to 6h (wait for weekly close)
    wp_6h = align_htf_to_ltf(prices, df_1w, wp)
    wr1_6h = align_htf_to_ltf(prices, df_1w, wr1)
    ws1_6h = align_htf_to_ltf(prices, df_1w, ws1)
    wr2_6h = align_htf_to_ltf(prices, df_1w, wr2)
    ws2_6h = align_htf_to_ltf(prices, df_1w, ws2)
    
    # Get daily data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(wp_6h[i]) or np.isnan(vol_ma_20_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 * 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20_6h[i]
        
        if position == 0:
            # Long: price above weekly R1 and volume spike
            if close[i] > wr1_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly S1 and volume spike
            elif close[i] < ws1_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly pivot or volume spike ends
            if close[i] < wp_6h[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly pivot or volume spike ends
            if close[i] > wp_6h[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals