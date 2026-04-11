#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels identify key support/resistance. Price breaking above
# resistance or below support with volume confirmation and trend alignment offers high-
# probability trades. Designed for low frequency (15-25 trades/year) to minimize fee drag
# and work in both bull and bear markets by following the higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla levels use previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid first value which will be NaN due to shift
    valid_idx = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
    
    # Camarilla levels
    # Resistance levels: R1, R2, R3, R4
    # Support levels: S1, S2, S3, S4
    # Formula: 
    # R4 = Close + (High - Low) * 1.5
    # R3 = Close + (High - Low) * 1.25
    # R2 = Close + (High - Low) * 1.166
    # R1 = Close + (High - Low) * 1.083
    # S1 = Close - (High - Low) * 1.083
    # S2 = Close - (High - Low) * 1.166
    # S3 = Close - (High - Low) * 1.25
    # S4 = Close - (High - Low) * 1.5
    
    hl_range = prev_high - prev_low
    r4 = prev_close + hl_range * 1.5
    r3 = prev_close + hl_range * 1.25
    r2 = prev_close + hl_range * 1.166
    r1 = prev_close + hl_range * 1.083
    s1 = prev_close - hl_range * 1.083
    s2 = prev_close - hl_range * 1.166
    s3 = prev_close - hl_range * 1.25
    s4 = prev_close - hl_range * 1.5
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        # Long when price breaks above R1 with volume spike and uptrend
        # Short when price breaks below S1 with volume spike and downtrend
        if (close[i] > r1_aligned[i] and volume_spike[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < s1_aligned[i] and volume_spike[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to midpoint or trend changes
        elif position == 1 and (close[i] < (r1_aligned[i] + s1_aligned[i]) / 2 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > (r1_aligned[i] + s1_aligned[i]) / 2 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals