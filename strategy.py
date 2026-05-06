#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 1h Camarilla R3 level AND 4h close > 4h EMA50 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 1h Camarilla S3 level AND 4h close < 4h EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 1h Camarilla pivot point (mean of high and low from prior 4h bar)
# Uses discrete sizing 0.20 to balance return and fee drag
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels provide robust support/resistance based on prior 4h bar's range
# 4h EMA50 filters for higher timeframe trend alignment
# Volume spike confirmation reduces false breakouts during low participation
# Session filter (08-20 UTC) avoids low-liquidity periods
# Works in both bull and bear markets by following the 4h trend

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Calculate 1h Camarilla levels and 4h EMA50 ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1h) < 1 or len(df_4h) < 50:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 1h timeframe using prior 4h bar
    # R3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    # S3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    # PP = (high_4h + low_4h + close_4h) / 3
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_val = df_4h['close'].values
    
    camarilla_r3 = close_4h_val + 1.1 * (high_4h - low_4h) / 2.0
    camarilla_s3 = close_4h_val - 1.1 * (high_4h - low_4h) / 2.0
    camarilla_pp = (high_4h + low_4h + close_4h_val) / 3.0
    
    # Calculate 4h EMA50 trend filter
    close_4h_series = pd.Series(close_4h_val)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price < S3 AND downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retests pivot point from above
            if close[i] <= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retests pivot point from below
            if close[i] >= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals