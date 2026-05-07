#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation.
# Uses 20-period Donchian channels on 6h, filtered by 1-week pivot direction (above/below pivot point)
# and confirmed by volume spikes. Works in bull/bear by aligning with higher timeframe structure.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).
name = "6h_DonchianBreakout_1wPivot_Volume"
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
    
    # Load 1w data ONCE for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot point and support/resistance levels
    # Standard pivot: (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Donchian channels (20-period) on 6h
    donch_period = 20
    dc_high = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    dc_low = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume spike detection (20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema > 0, volume / vol_ema, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for Donchian and volume
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine bias from weekly pivot: above pivot = bullish bias, below = bearish bias
        bullish_bias = close[i] > pivot_1w_aligned[i]
        bearish_bias = close[i] < pivot_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume spike in bullish bias
            long_condition = (close[i] > dc_high[i]) and vol_spike[i] and bullish_bias
            # Short breakdown: price breaks below Donchian low with volume spike in bearish bias
            short_condition = (close[i] < dc_low[i]) and vol_spike[i] and bearish_bias
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below Donchian high or bias turns bearish
            if (close[i] < dc_high[i]) or (not bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above Donchian low or bias turns bullish
            if (close[i] > dc_low[i]) or (not bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals