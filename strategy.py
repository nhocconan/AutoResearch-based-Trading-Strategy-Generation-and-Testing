#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels provide intraday support/resistance; breakout of R3/S3 indicates strong momentum
# 4h EMA50 ensures we trade with the higher timeframe trend
# Volume confirmation filters out false breakouts
# Session filter (08-20 UTC) reduces noise during low-activity periods
# Discrete sizing 0.20 targets 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 trend filter from prior completed 4h bar
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_shifted = np.roll(ema50_4h, 1)
    ema50_4h_shifted[0] = np.nan
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_shifted)
    
    # Get 1d data for Camarilla pivot calculation (using prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior completed 1d bar
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using the standard Camarilla formula: R4 = close + 1.1*(high-low)*1.1/2, but we use R3/S3
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Simplified: R3 = close + 1.1*(high-low)*0.275, S3 = close - 1.1*(high-low)*0.275
    # Actually standard Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_multiplier = 1.1 / 4.0
    camarilla_range = (high_1d - low_1d) * camarilla_multiplier
    r3_1d = close_1d + camarilla_range
    s3_1d = close_1d - camarilla_range
    
    # Shift to get prior completed 1d bar values
    r3_1d_shifted = np.roll(r3_1d, 1)
    s3_1d_shifted = np.roll(s3_1d, 1)
    r3_1d_shifted[0] = np.nan
    s3_1d_shifted[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_shifted)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 4h EMA50 uptrend AND volume spike
            if close[i] > r3_1d_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND 4h EMA50 downtrend AND volume spike
            elif close[i] < s3_1d_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 OR 4h EMA50 turns down
            if close[i] < s3_1d_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R3 OR 4h EMA50 turns up
            if close[i] > r3_1d_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals