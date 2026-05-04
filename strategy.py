#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla pivots provide intraday support/resistance levels based on prior day's range
# 4h EMA50 establishes medium-term trend direction to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA volume) confirms institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Discrete sizing 0.20 targets 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Works in bull markets (breakouts with trend alignment) and bear markets (breakouts with trend alignment)

name = "1h_Camarilla_Pivot_Breakout_4hEMA50_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivot calculation (prior completed day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for prior day calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior completed 1d bar
    # Camarilla R1, R3, S1, S3 based on prior day's range
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Camarilla formulas: Range = prior_high - prior_low
    # R4 = prior_close + Range * 1.1/2
    # R3 = prior_close + Range * 1.1/4
    # R2 = prior_close + Range * 1.1/6
    # R1 = prior_close + Range * 1.1/12
    # S1 = prior_close - Range * 1.1/12
    # S2 = prior_close - Range * 1.1/6
    # S3 = prior_close - Range * 1.1/4
    # S4 = prior_close - Range * 1.1/2
    camarilla_range = prior_high - prior_low
    camarilla_r3 = prior_close + camarilla_range * 1.1 / 4
    camarilla_s3 = prior_close - camarilla_range * 1.1 / 4
    camarilla_r1 = prior_close + camarilla_range * 1.1 / 12
    camarilla_s1 = prior_close - camarilla_range * 1.1 / 12
    
    # Align HTF Camarilla levels to 1h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND above 4h EMA50 AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND below 4h EMA50 AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla R1 OR below 4h EMA50
            if close[i] < camarilla_r1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Camarilla S1 OR above 4h EMA50
            if close[i] > camarilla_s1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals