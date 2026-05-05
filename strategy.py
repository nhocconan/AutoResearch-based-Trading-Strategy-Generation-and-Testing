#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h volume confirmation and 1d EMA200 trend filter
# Long when price breaks above 1h Camarilla R1 level AND 4h volume > 2.0x 20-period average AND close > 1d EMA200
# Short when price breaks below 1h Camarilla S1 level AND 4h volume > 2.0x 20-period average AND close < 1d EMA200
# Exit when price crosses 1h Camarilla pivot point (mean reversion)
# Uses 1h primary timeframe with 4h HTF for volume confirmation and 1d HTF for trend filter
# Session filter (08-20 UTC) to reduce noise trades
# Discrete sizing (0.20) to limit fee drag and manage drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe

name = "1h_Camarilla_R1S1_Breakout_4hVolume_1dEMA200"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h volume spike filter
    vol_4h = df_4h['volume'].values
    if len(vol_4h) >= 20:
        vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        volume_filter_4h = vol_4h > (2.0 * vol_ma_20)
    else:
        volume_filter_4h = np.zeros(len(df_4h), dtype=bool)
    
    # Align 4h volume filter to 1h timeframe
    volume_filter_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_filter_4h)
    
    # Get 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 1h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 1h data ONCE before loop for Camarilla levels
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    camarilla_r1 = close_1h + (1.1 * (high_1h - low_1h) / 12)
    camarilla_s1 = close_1h - (1.1 * (high_1h - low_1h) / 12)
    camarilla_pivot = (high_1h + low_1h + close_1h) / 3  # Standard pivot point
    
    # Align Camarilla levels to 1h timeframe (same df_1h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1h, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_filter_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND volume spike AND above 1d EMA200
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter_4h_aligned[i] and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND volume spike AND below 1d EMA200
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter_4h_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals