#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 breakouts with 1-week EMA50 trend filter and volume confirmation (>1.8x 20-day MA). R1/S1 levels provide tighter breakout thresholds than R4/S4, increasing trade frequency while maintaining edge through 1-week trend alignment. Designed for 1d timeframe to minimize fee drag and maximize test generalization. Volume spike filter reduces whipsaws. Works in bull/bear markets by following 1-week trend while using Camarilla structure for precise entries.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Previous day's OHLC for Camarilla levels (using 1d for structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 (tighter breakout levels)
    rng = high_1d - low_1d
    camarilla_r1 = close_1d_vals + (rng * 1.1 / 12)  # R1 level
    camarilla_s1 = close_1d_vals - (rng * 1.1 / 12)  # S1 level
    
    # Align Camarilla levels to 1d timeframe (no alignment needed as both are 1d)
    camarilla_r1_aligned = camarilla_r1  # Already aligned to 1d
    camarilla_s1_aligned = camarilla_s1  # Already aligned to 1d
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 50 for 1w EMA, 0 for camarilla)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1w trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla R1/S1 in trend direction with volume spike
        long_entry = (close_val > camarilla_r1_val) and bullish_1w and vol_spike
        short_entry = (close_val < camarilla_s1_val) and bearish_1w and vol_spike
        
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
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point or not bullish_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point or not bearish_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0