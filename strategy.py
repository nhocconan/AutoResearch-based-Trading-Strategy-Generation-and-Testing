#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 1d for structure, 1d EMA34 for trend alignment (works in bull/bear)
# Volume spike (>1.5x 20-bar average) confirms institutional interest and reduces false breakouts
# Discrete sizing 0.25 to limit fee drag; target 50-150 total trades over 4 years
# Proven pattern: Camarilla R3/S3 breaks with volume/volume confirmation work on BTC/ETH in both bull/bear

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # Using previous day's high/low/close for current day's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range
    s3 = prev_close_1d - 1.1 * camarilla_range
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume spike filter (1.5x 20-bar average)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma_20)
    
    # Align HTF indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend (price > EMA34) AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend (price < EMA34) AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests S3 from above (trend reversal)
            if close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests R3 from below (trend reversal)
            if close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals