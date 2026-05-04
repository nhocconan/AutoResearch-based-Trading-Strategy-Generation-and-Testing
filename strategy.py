#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spike confirmation.
# Uses 4h/1d for signal direction (structure/trend) and 1h only for entry timing precision to minimize fee drag.
# Session filter (08-20 UTC) reduces noise trades. Position size fixed at 0.20 for risk control.
# Designed for 15-30 trades/year to avoid fee drag while capturing breakout continuations in both bull and bear markets.

name = "1h_Camarilla_R3S3_1dEMA34_VolumeSpike_Session"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate 4h Camarilla pivot levels from prior completed 4h bar
    R3_4h = close_4h + 1.125 * (high_4h - low_4h)
    S3_4h = close_4h - 1.125 * (high_4h - low_4h)
    R3_4h_shifted = np.roll(R3_4h, 1)
    S3_4h_shifted = np.roll(S3_4h, 1)
    R3_4h_shifted[0] = np.nan
    S3_4h_shifted[0] = np.nan
    R3_4h_aligned = align_htf_to_ltf(prices, df_4h, R3_4h_shifted)
    S3_4h_aligned = align_htf_to_ltf(prices, df_4h, S3_4h_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R3_4h_aligned[i]) or
            np.isnan(S3_4h_aligned[i]) or
            np.isnan(vol_ema_20[i])):
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
            # Long conditions: price breaks above R3_4h AND above 1d EMA34 AND volume spike
            if close[i] > R3_4h_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3_4h AND below 1d EMA34 AND volume spike
            elif close[i] < S3_4h_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S3_4h OR below 1d EMA34
            if close[i] < S3_4h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R3_4h OR above 1d EMA34
            if close[i] > R3_4h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals