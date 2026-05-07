#!/usr/bin/env python3
# 1h_Camarilla_R3S3_Breakout_4hTrend_Volume
# Hypothesis: Uses 1d Camarilla R3/S3 as breakout levels, filtered by 4h EMA50 trend and volume spikes on 1h.
# The 4h EMA50 provides trend direction, while 1d Camarilla levels offer high-probability breakout zones.
# Volume spikes confirm breakout strength. Trades only during 08-20 UTC session to reduce noise.
# Designed for 1h timeframe with strict entry conditions to limit trades (target: 15-37/year).
# Works in bull/bear markets via trend filter and breakout logic.

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_1h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_1h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume spike on 1h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(camarilla_r3_1h[i]) or np.isnan(camarilla_s3_1h[i]) or 
            np.isnan(ema_50_4h_1h[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 + above 4h EMA50 + volume spike
            if close[i] > camarilla_r3_1h[i] and close[i] > ema_50_4h_1h[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 + below 4h EMA50 + volume spike
            elif close[i] < camarilla_s3_1h[i] and close[i] < ema_50_4h_1h[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Price closes below Camarilla S3 or below 4h EMA50
            if close[i] < camarilla_s3_1h[i] or close[i] < ema_50_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price closes above Camarilla R3 or above 4h EMA50
            if close[i] > camarilla_r3_1h[i] or close[i] > ema_50_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals