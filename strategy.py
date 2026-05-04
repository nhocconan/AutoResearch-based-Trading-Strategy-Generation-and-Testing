#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 4h chart for structure, 4h EMA50 for trend filter, and volume spike for confirmation.
# Designed for 15-37 trades/year on 1h timeframe to minimize fee drag. Works in bull markets via breakout continuations 
# and in bear markets via breakdown continuations. Session filter (08-20 UTC) reduces noise trades.
# The 4h EMA50 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.
# BTC and ETH are primary targets.

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_SessionFilter"
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
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA50 trend filter from prior completed 4h bar
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_shifted = np.roll(ema50_4h, 1)
    ema50_4h_shifted[0] = np.nan
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_shifted)
    
    # Calculate Camarilla levels (R1, S1) from prior completed 4h bar
    # Camarilla: R1 = close + 1.1*(high-low)/6, S1 = close - 1.1*(high-low)/6
    camarilla_range = high_4h - low_4h
    r1 = close_4h + 1.1 * camarilla_range / 6
    s1 = close_4h - 1.1 * camarilla_range / 6
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND above 4h EMA50 AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S1 AND below 4h EMA50 AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 OR below 4h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R1 OR above 4h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals