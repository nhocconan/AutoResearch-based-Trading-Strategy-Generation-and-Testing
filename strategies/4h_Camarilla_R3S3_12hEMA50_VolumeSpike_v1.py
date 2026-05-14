#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses 12h EMA for stronger trend alignment than 1d to reduce whipsaw in bear markets
# Volume spike (>2x 20-bar average) confirms institutional participation
# Discrete sizing 0.25 to limit fee drag; target 75-200 trades over 4 years
# Camarilla pivots work in ranging markets; EMA filter adds directional bias

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_4h) < 1 or len(df_12h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_12h = df_12h['close'].values
    
    # Calculate 4h Camarilla levels from prior 4h bar
    prior_high = np.roll(high_4h, 1)
    prior_low = np.roll(low_4h, 1)
    prior_close = np.roll(close_4h, 1)
    prior_high[0] = np.nan  # Invalidate first rolled value
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 2.0
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 2.0
    
    # Calculate 12h EMA50 trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend (price > EMA50) AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend (price < EMA50) AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests pivot from above
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests pivot from below
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals