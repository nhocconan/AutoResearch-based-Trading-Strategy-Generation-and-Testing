#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above 12h Camarilla R3 level AND 1d close > 1d EMA50 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 12h Camarilla S3 level AND 1d close < 1d EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 12h Camarilla midpoint (mean of R3 and S3)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla pivot levels provide clear structure for breakouts in both bull and bear markets
# 1d EMA50 filters for higher timeframe trend alignment (proven BTC/ETH edge from research)
# Volume spike confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 1d trend

name = "12h_Camarilla_R3S3_1dEMA50_VolumeSpike_v1"
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
    
    # Calculate 12h Camarilla and 1d EMA50 ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 5 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Camarilla levels (based on previous day's range)
    # Camarilla R3 = close + 1.1 * (high - low) / 2
    # Camarilla S3 = close - 1.1 * (high - low) / 2
    # Camarilla midpoint = close
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = np.nan
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    camarilla_r3 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 2.0
    camarilla_s3 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 2.0
    camarilla_mid = prev_close_12h  # midpoint is previous close
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests midpoint from above
            if close[i] <= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests midpoint from below
            if close[i] >= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals