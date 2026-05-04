#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from 6h chart for structure, 12h EMA50 for trend filter,
# and volume spike for confirmation. Designed for 12-37 trades/year (50-150 total over 4 years)
# to minimize fee drag. Works in bull markets via upward breakouts at R4 and in bear markets
# via downward breakdowns at S4, with fading at R3/S3 in ranging markets.
# The 12h EMA50 provides a smooth trend filter that avoids whipsaw.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_TrendFilter"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_shifted = np.roll(ema50_12h, 1)
    ema50_12h_shifted[0] = np.nan
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_shifted)
    
    # Calculate Camarilla levels from prior completed 6h bar
    # Camarilla: based on previous day's (6h bar's) high, low, close
    lookback = 1
    prev_high = pd.Series(high).shift(lookback).values
    prev_low = pd.Series(low).shift(lookback).values
    prev_close = pd.Series(close).shift(lookback).values
    
    # Camarilla levels
    R4 = prev_close + (prev_high - prev_low) * 1.500
    R3 = prev_close + (prev_high - prev_low) * 1.250
    S3 = prev_close - (prev_high - prev_low) * 1.250
    S4 = prev_close - (prev_high - prev_low) * 1.500
    
    # Shift to ensure we only use completed bars
    R4_shifted = np.roll(R4, 1)
    R3_shifted = np.roll(R3, 1)
    S3_shifted = np.roll(S3, 1)
    S4_shifted = np.roll(S4, 1)
    R4_shifted[0] = np.nan
    R3_shifted[0] = np.nan
    S3_shifted[0] = np.nan
    S4_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(R4_shifted[i]) or
            np.isnan(R3_shifted[i]) or
            np.isnan(S3_shifted[i]) or
            np.isnan(S4_shifted[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 AND above 12h EMA50 AND volume spike
            if close[i] > R4_shifted[i] and close[i] > ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S4 AND below 12h EMA50 AND volume spike
            elif close[i] < S4_shifted[i] and close[i] < ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
            # Long fade: price rejects at S3 (reverses up) AND above 12h EMA50 AND volume spike
            elif low[i] <= S3_shifted[i] and close[i] > S3_shifted[i] and close[i] > ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short fade: price rejects at R3 (reverses down) AND below 12h EMA50 AND volume spike
            elif high[i] >= R3_shifted[i] and close[i] < R3_shifted[i] and close[i] < ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 OR below 12h EMA50
            if close[i] < S3_shifted[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 OR above 12h EMA50
            if close[i] > R3_shifted[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals