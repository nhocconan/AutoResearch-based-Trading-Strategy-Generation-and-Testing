#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike
# Uses 4h/1d for signal direction, 1h only for entry timing precision.
# Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years.
# Works in bull via upward breaks at R3/R4 and in bear via downward breaks at S3/S4.
# Volume confirmation filters false breakouts.

name = "1h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "1h"
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
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    R3_4h = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h)
    S3_4h = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h)
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3_4h)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3_4h)
    
    # Volume confirmation: 20-period EMA on 1h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 in uptrend alignment with volume spike
            if close[i] > R3_aligned[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 in downtrend alignment with volume spike
            elif close[i] < S3_aligned[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or loses uptrend alignment
            if close[i] < S3_aligned[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 or loses downtrend alignment
            if close[i] > R3_aligned[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals