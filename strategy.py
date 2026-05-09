#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (EMA50) and volume spike.
# R3/S3 levels are more extreme than R1/S1, reducing false breakouts.
# Volume spike confirms institutional interest. EMA50 filter ensures trend alignment.
# Works in bull (breakouts with trend) and bear (mean reversion from extremes).
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, trend, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R3, S3) - more extreme than R1/S1
    r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 2.0 * 20-day average (spike)
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (vol_ma * 2.0)
    
    # Align all to 4h
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or
            np.isnan(ema50_1d_4h[i]) or np.isnan(volume_spike_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_4h[i]
        s3_val = s3_4h[i]
        trend = ema50_1d_4h[i]
        vol_spike = volume_spike_4h[i]
        
        if position == 0:
            # Enter long: break above R3 with volume spike and above trend
            if close[i] > r3_val and close[i] > trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with volume spike and below trend
            elif close[i] < s3_val and close[i] < trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 (mean reversion to center)
            if close[i] < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 (mean reversion to center)
            if close[i] > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals