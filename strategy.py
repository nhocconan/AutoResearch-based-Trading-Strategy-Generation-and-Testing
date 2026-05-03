#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves. Using 12h EMA50 as trend filter
# ensures we trade with the higher timeframe trend. Volume confirmation filters false breakouts.
# Designed for 20-50 trades/year on 4h to minimize fee drag while maintaining edge in both bull and bear markets.

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian(20) channels
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup for Donchian
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above/below Donchian channels with volume spike
        breakout_long = close[i] > high_rolling_max[i-1] and volume_spike[i]
        breakout_short = close[i] < low_rolling_min[i-1] and volume_spike[i]
        
        if position == 0:
            # Long: break above upper Donchian in 12h uptrend with volume spike
            if breakout_long and ema_50_12h_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian in 12h downtrend with volume spike
            elif breakout_short and ema_50_12h_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower Donchian or loses 12h uptrend
            if close[i] < low_rolling_min[i] or ema_50_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper Donchian or loses 12h downtrend
            if close[i] > high_rolling_max[i] or ema_50_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals