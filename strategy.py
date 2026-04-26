#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R4/S4 breakouts with 12h EMA50 trend filter and volume confirmation (>1.8x 20-bar MA). R4/S4 represent stronger breakout levels than R3/S3, reducing false signals while capturing momentum. Designed for 6h timeframe to balance trade frequency and edge. Works in bull/bear markets by following 12h trend while using Camarilla structure for entries. Volume spike filter reduces whipsaws. Target: 12-30 trades/year (50-120 total over 4 years).
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
    
    # Load 12h data ONCE before loop for HTF filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Previous day's OHLC for Camarilla levels (using 1d for structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R4, S4 (stronger breakout levels)
    rng = high_1d - low_1d
    camarilla_r4 = close_1d_vals + (rng * 1.1 / 2)  # R4 level
    camarilla_s4 = close_1d_vals - (rng * 1.1 / 2)  # S4 level
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 50 for 12h EMA, 1 for camarilla)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r4_val = camarilla_r4_aligned[i]
        camarilla_s4_val = camarilla_s4_aligned[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 12h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_12h = close_val > ema_50_val
        bearish_12h = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla R4/S4 in trend direction with volume spike
        long_entry = (close_val > camarilla_r4_val) and bullish_12h and vol_spike
        short_entry = (close_val < camarilla_s4_val) and bearish_12h and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r4_val + camarilla_s4_val) / 2
            if close_val < mid_point or not bullish_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r4_val + camarilla_s4_val) / 2
            if close_val > mid_point or not bearish_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0