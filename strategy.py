#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA50 trend filter and volume confirmation (1.8x)
# Long when price breaks above R3 AND price > 4h EMA50 AND volume > 1.8x 20-period average AND session (08-20 UTC)
# Short when price breaks below S3 AND price < 4h EMA50 AND volume > 1.8x 20-period average AND session (08-20 UTC)
# Exit when price reverts to Camarilla pivot point (PP) OR 4h EMA50 filter reverses
# Uses Camarilla levels for precise intraday structure + volume confirmation to reduce false signals
# 4h EMA50 provides higher timeframe trend filter effective in both bull and bear markets
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Timeframe: 1h (primary), HTF: 4h

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_1.8x_SessionFilter"
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
    
    # Get 4h data ONCE before loop for Camarilla and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar
    camarilla_high = np.roll(high_4h, 1)  # previous 4h bar high
    camarilla_low = np.roll(low_4h, 1)    # previous 4h bar low
    camarilla_close = np.roll(close_4h, 1) # previous 4h bar close
    
    # Calculate Camarilla levels
    camarilla_range = camarilla_high - camarilla_low
    r3 = camarilla_close + 1.1 * camarilla_range
    s3 = camarilla_close - 1.1 * camarilla_range
    pp = (camarilla_high + camarilla_low + camarilla_close) / 3.0
    
    # Calculate 4h EMA(50)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation on 1h (threshold: 1.8x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND price > EMA50 AND volume spike AND in session
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i] and 
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND price < EMA50 AND volume spike AND in session
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i] and 
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price reverts to PP OR price < EMA50 (trend weakening)
            if close[i] < pp_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price reverts to PP OR price > EMA50 (trend weakening)
            if close[i] > pp_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals