#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h KAMA trend filter and volume confirmation.
# Donchian(20) breakout captures breakout momentum. KAMA on 12h adapts to market regime (trending/ranging).
# Volume > 1.5x average confirms institutional interest. Designed for low trade frequency (<30/year) to minimize fee drag.
name = "4h_Donchian20_12hKAMA_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close
    close_12h = df_12h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.abs(np.diff(close_12h))
    er = np.zeros_like(close_12h)
    er[1:] = change[1:] / np.where(volatility[1:] == 0, 1, volatility[1:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.maximum.accumulate(high)
    donchian_low = np.minimum.accumulate(low)
    # Reset every 20 periods
    for i in range(20, len(donchian_high)):
        if i % 20 == 0:
            donchian_high[i] = high[i]
            donchian_low[i] = low[i]
        else:
            donchian_high[i] = max(donchian_high[i-1], high[i])
            donchian_low[i] = min(donchian_low[i-1], low[i])
    # Alternative: use rolling window
    donchian_high = np.array([np.max(high[max(0, i-19):i+1]) for i in range(len(high))])
    donchian_low = np.array([np.min(low[max(0, i-19):i+1]) for i in range(len(low))])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(kama_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_12h = kama_12h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > upper Donchian AND price > 12h KAMA (uptrend) AND volume > 1.5x average
            if close[i] > upper and close[i] > kama_12h and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < lower Donchian AND price < 12h KAMA (downtrend) AND volume > 1.5x average
            elif close[i] < lower and close[i] < kama_12h and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < lower Donchian OR trend reverses (price < 12h KAMA)
            if close[i] < lower or close[i] < kama_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > upper Donchian OR trend reverses (price > 12h KAMA)
            if close[i] > upper or close[i] > kama_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals