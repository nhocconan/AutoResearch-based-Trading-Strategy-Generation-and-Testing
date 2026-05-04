#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Donchian breakouts capture strong momentum moves. 1w EMA50 ensures alignment with weekly trend.
# Volume confirmation filters low-conviction breakouts. Designed for 7-25 trades/year on 1d to minimize fee drag.
# Works in bull markets via long on upper band breakout in uptrend and in bear markets via short on lower band breakout in downtrend.

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 15m timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # Get 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian band AND 1w uptrend AND volume spike
            if (close[i] > upper_band_aligned[i] and   # Break above upper band
                close[i] > ema_50_aligned[i] and       # 1w uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian band AND 1w downtrend AND volume spike
            elif (close[i] < lower_band_aligned[i] and   # Break below lower band
                  close[i] < ema_50_aligned[i] and       # 1w downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below midpoint OR 1w trend turns down
            midpoint = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
            if close[i] < midpoint or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midpoint OR 1w trend turns up
            midpoint = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
            if close[i] > midpoint or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals