#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour KAMA(10) trend filter and volume confirmation.
# The 12-hour KAMA adapts to market efficiency, reducing noise in choppy markets while following trends.
# The Donchian(20) breakout captures momentum in the direction of the 12-hour KAMA trend.
# Volume > 1.5x the 20-period average confirms institutional participation and reduces false breakouts.
# Exit occurs when price crosses the 12-hour KAMA. This combination targets 20-40 trades per year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h KAMA(10) for trend filter
    kama_len = 10
    if len(df_12h) < kama_len:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(df_12h['close'], prepend=df_12h['close'].iloc[0]))
    volatility = np.abs(np.diff(df_12h['close'])).rolling(window=kama_len, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(df_12h['close'], dtype=float)
    kama[0] = df_12h['close'].iloc[0]
    for i in range(1, len(df_12h)):
        kama[i] = kama[i-1] + sc[i] * (df_12h['close'].iloc[i] - kama[i-1])
    
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Donchian channel (20 periods) on 4h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(kama_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h KAMA
        above_kama = close[i] > kama_12h_aligned[i]
        below_kama = close[i] < kama_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + above 12h KAMA + volume
            if (close[i] > dc_upper[i] and 
                above_kama and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + below 12h KAMA + volume
            elif (close[i] < dc_lower[i] and 
                  below_kama and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 12h KAMA
            if close[i] < kama_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 12h KAMA
            if close[i] > kama_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_KAMA10_Donchian_Volume_v1"
timeframe = "4h"
leverage = 1.0