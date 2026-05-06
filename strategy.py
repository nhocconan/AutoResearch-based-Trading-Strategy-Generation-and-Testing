#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 12h Camarilla R3 level AND 1d close > 1d EMA34 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 12h Camarilla S3 level AND 1d close < 1d EMA34 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 12h Camarilla pivot point (mean of high and low from prior 1d bar)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels provide robust support/resistance based on prior day's range
# 1d EMA34 filters for higher timeframe trend alignment
# Volume spike confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 1d trend

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels and 1d EMA34 ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 1 or len(df_1d) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 12h timeframe using prior 1d bar
    # R3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    # S3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    # PP = (high_1d + low_1d + close_1d) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_val = df_1d['close'].values
    
    camarilla_r3 = close_1d_val + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3 = close_1d_val - 1.1 * (high_1d - low_1d) / 2.0
    camarilla_pp = (high_1d + low_1d + close_1d_val) / 3.0
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d_val)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
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