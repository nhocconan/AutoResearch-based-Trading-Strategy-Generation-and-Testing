#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h chart with weekly pivot points (calculated from prior week)
# and volume confirmation. Price breaking weekly R1 or S1 with volume > 2x mean
# indicates institutional interest. Trend filter: price above/below 100-period EMA
# on 6s chart to avoid counter-trend trades. Works in bull/bear by capturing
# breakouts from key weekly levels.
# Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data (HTF for pivot points) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Pivot Points (standard calculation) ===
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    # First bar: use current values
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    prev_range = prev_high_1w - prev_low_1w
    
    # R1 and S1 at 1.0 * range (standard)
    r1 = pivot_point + prev_range
    s1 = pivot_point - prev_range
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === 6h EMA100 for trend filter ===
    ema_100 = pd.Series(close).ewm(span=100, min_periods=100, adjust=False).mean().values
    
    # === Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_100[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_100_val = ema_100[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 (stop) or at 1.5x R1 (take profit)
            if price < s1_val or price > r1_val * 1.5:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 (stop) or at 0.5x S1 (take profit)
            if price > r1_val or price < s1_val * 0.5:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume AND above EMA100 (uptrend)
            if (price > r1_val) and (price > ema_100_val) and (vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume AND below EMA100 (downtrend)
            elif (price < s1_val) and (price < ema_100_val) and (vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R1_S1_EMA100_Volume"
timeframe = "6h"
leverage = 1.0