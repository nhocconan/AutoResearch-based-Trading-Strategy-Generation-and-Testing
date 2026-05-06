#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend filter and volume spike confirmation
# Long when price breaks above 6h Camarilla R3 AND 12h close > 12h EMA50 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 6h Camarilla S3 AND 12h close < 12h EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 6h Camarilla pivot point (mean of H/L/C from prior 6h bar)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla levels provide precise intraday support/resistance for breakouts
# 12h EMA50 filters for higher timeframe trend alignment (proven BTC/ETH edge from research)
# Volume spike confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 12h trend

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Calculate 6h Camarilla levels and 12h EMA50 ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_6h) < 2 or len(df_12h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    close_12h = df_12h['close'].values
    
    # Calculate 6h Camarilla levels (R3, S3, pivot) from prior bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2, pivot = (high+low+close)/3
    # Use prior bar's values to avoid look-ahead
    high_6h_prev = np.roll(high_6h, 1)
    low_6h_prev = np.roll(low_6h, 1)
    close_6h_prev = np.roll(close_6h, 1)
    high_6h_prev[0] = np.nan
    low_6h_prev[0] = np.nan
    close_6h_prev[0] = np.nan
    
    camarilla_pivot = (high_6h_prev + low_6h_prev + close_6h_prev) / 3.0
    camarilla_r3 = close_6h_prev + 1.1 * (high_6h_prev - low_6h_prev) / 2.0
    camarilla_s3 = close_6h_prev - 1.1 * (high_6h_prev - low_6h_prev) / 2.0
    
    # Calculate 12h EMA50 trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed bars)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests pivot from above
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests pivot from below
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals