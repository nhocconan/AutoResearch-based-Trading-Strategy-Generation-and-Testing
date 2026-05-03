#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivots from 1d provide robust support/resistance levels.
# Breakout at R3 (long) or S3 (short) with volume spike (2.0x 20-period EMA) and trend filter (price > 1d EMA34 for long, < for short).
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
# Works in both bull and bear markets by using 1d EMA34 for trend alignment.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate previous day's Camarilla levels
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Use previous day's H, L, C to avoid look-ahead
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Previous day's values
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        camarilla_r3[i] = c + ((h - l) * 1.1 / 4)
        camarilla_s3[i] = c - ((h - l) * 1.1 / 4)
    
    # Align 1d indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and price > 1d EMA34 (uptrend)
            if close[i] > camarilla_r3_aligned[i] and volume_spike and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and price < 1d EMA34 (downtrend)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or loses uptrend (price < 1d EMA34)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or loses downtrend (price > 1d EMA34)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals