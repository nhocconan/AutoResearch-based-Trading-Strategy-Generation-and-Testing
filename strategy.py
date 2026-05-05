#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (1.5x)
# Long when price breaks above upper Donchian AND price > 1w EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below lower Donchian AND price < 1w EMA50 AND volume > 1.5x 20-period average
# Exit when price reverts to the opposite Donchian band OR 1w EMA50 filter reverses
# Uses 1d timeframe with 1w HTF for lower trade frequency (target: 30-100 total over 4 years)
# Donchian channels provide robust price structure from 20-day lookback
# Volume confirmation reduces false breakouts
# 1w EMA50 offers strong trend filter effective in both bull and bear markets
# Designed for minimal fee drag with discrete position sizing (0.25)

name = "1d_Donchian20_1wEMA50_VolumeSpike_1.5x"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicator to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 1d (20-period)
    if len(high) >= 20:
        upper_donch = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_donch = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper_donch = np.full(n, np.nan)
        lower_donch = np.full(n, np.nan)
    
    # Volume confirmation on 1d (threshold: 1.5x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > EMA50 AND volume spike
            if (close[i] > upper_donch[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < EMA50 AND volume spike
            elif (close[i] < lower_donch[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to lower Donchian OR price < EMA50 (trend weakening)
            if close[i] < lower_donch[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to upper Donchian OR price > EMA50 (trend weakening)
            if close[i] > upper_donch[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals