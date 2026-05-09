#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    
    # Get 12h data for EMA50 and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    # 12h volume average for volume spike filter
    vol_series = pd.Series(df_12h['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = df_12h['volume'].values > (vol_ma * 2.0)  # Volume > 2x 20-period average
    
    # Align 12h indicators to 4h
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_spike_4h = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Donchian(20) channels on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_4h[i]) or np.isnan(volume_spike_4h[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian with volume spike and above 12h EMA50
            if close[i] > upper[i] and volume_spike_4h[i] and close[i] > ema50_12h_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian with volume spike and below 12h EMA50
            elif close[i] < lower[i] and volume_spike_4h[i] and close[i] < ema50_12h_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches lower Donchian (mean reversion)
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches upper Donchian (mean reversion)
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals