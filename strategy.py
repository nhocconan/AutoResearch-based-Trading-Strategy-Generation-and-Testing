#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 1d timeframe for structure, 1d EMA34 for trend filter,
# and volume spike for confirmation. Designed for 12-30 trades/year to minimize fee drag.
# Works in bull markets via R4/S4 breakout continuations and in bear markets via mean reversion at R3/S3.
# The 1d EMA34 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    # Camarilla: P = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + (range_1d * 1.1 / 2.0)
    s3_1d = close_1d - (range_1d * 1.1 / 2.0)
    r4_1d = close_1d + (range_1d * 1.1)
    s4_1d = close_1d - (range_1d * 1.1)
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r3_1d_shifted = np.roll(r3_1d, 1)
    s3_1d_shifted = np.roll(s3_1d, 1)
    r4_1d_shifted = np.roll(r4_1d, 1)
    s4_1d_shifted = np.roll(s4_1d, 1)
    r3_1d_shifted[0] = np.nan
    s3_1d_shifted[0] = np.nan
    r4_1d_shifted[0] = np.nan
    s4_1d_shifted[0] = np.nan
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_shifted)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4 OR mean reversion at S3 with volume confirmation
            # Only take long if price is above 1d EMA34 (uptrend bias)
            if close[i] > r4_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Mean reversion long at S3 in uptrend
            elif close[i] < s3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S4 OR mean reversion at R3 with volume confirmation
            # Only take short if price is below 1d EMA34 (downtrend bias)
            elif close[i] < s4_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
            # Mean reversion short at R3 in downtrend
            elif close[i] > r3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 OR above R4 (take profit at extreme levels)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 OR below S4 (take profit at extreme levels)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals