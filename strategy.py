#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakouts with volume confirmation
# and 1d trend filter (1d EMA 50). Breakouts above upper Donchian(20) or below lower
# Donchian(20) on the 4h timeframe, confirmed by volume > 1.5x 20-period average
# and aligned with 1d trend (price above/below 1d EMA 50). Uses 1h only for precise
# entry timing. Designed to work in both bull and bear markets by capturing
# directional breaks with volume confirmation. Target: 60-150 total trades over 4 years
# (15-37/year) with position size 0.20.

name = "1h_Donchian20_4hBreakout_Volume_1dTrend"
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
    
    # Calculate 4h Donchian channel (20-period) ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian upper and lower bands
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Rolling max/min for Donchian channels
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate 1d EMA 50 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-period average (moderate threshold to control trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 4h upper Donchian with volume and 1d uptrend
            if close[i] > upper_20_aligned[i] and volume_filter[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short breakout: price breaks below 4h lower Donchian with volume and 1d downtrend
            elif close[i] < lower_20_aligned[i] and volume_filter[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h lower Donchian (failed breakout) or reaches opposite band
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h upper Donchian (failed breakdown) or reaches opposite band
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals