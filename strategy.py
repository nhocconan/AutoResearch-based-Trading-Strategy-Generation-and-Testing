#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels derived from prior 1d OHLC: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
# Breakout above R3 with uptrend EMA34 and volume spike = long entry
# Breakdown below S3 with downtrend EMA34 and volume spike = short entry
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate Camarilla R3 and S3 levels from prior completed 1d bar
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Shift to use only completed 1d bar (no look-ahead)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above R3 AND 1d EMA34 uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below S3 AND 1d EMA34 downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 OR below EMA34
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S3 OR above EMA34
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals