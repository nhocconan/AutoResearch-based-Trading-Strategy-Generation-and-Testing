#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above 6h Camarilla R3 level AND 1w close > 1w EMA50 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 6h Camarilla S3 level AND 1w close < 1w EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 6h Camarilla pivot point (mean of high and low from prior 1w bar)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla levels provide robust support/resistance based on prior week's range
# 1w EMA50 filters for higher timeframe trend alignment
# Volume spike confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 1w trend

name = "6h_Camarilla_R3S3_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla levels and 1w EMA50 ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_6h) < 1 or len(df_1w) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_val = df_1w['close'].values
    
    # Calculate Camarilla levels for 6h timeframe using prior 1w bar
    # R3 = close_1w + 1.1 * (high_1w - low_1w) / 2
    # S3 = close_1w - 1.1 * (high_1w - low_1w) / 2
    # PP = (high_1w + low_1w + close_1w) / 3
    camarilla_r3 = close_1w_val + 1.1 * (high_1w - low_1w) / 2.0
    camarilla_s3 = close_1w_val - 1.1 * (high_1w - low_1w) / 2.0
    camarilla_pp = (high_1w + low_1w + close_1w_val) / 3.0
    
    # Calculate 1w EMA50 trend filter
    close_1w_series = pd.Series(close_1w_val)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests pivot point from above
            if close[i] <= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests pivot point from below
            if close[i] >= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals