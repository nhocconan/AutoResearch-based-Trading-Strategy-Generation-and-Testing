#!/usr/bin/env python3
"""
1d_ThreeBarReversal_LiquiditySweep
Hypothesis: Capture institutional reversal patterns where price sweeps liquidity (equal highs/lows) and reverses with a three-bar reversal pattern. This works in both bull and bear markets as it targets exhaustion moves. Uses 1d timeframe to limit trades and avoid fee drag. Confirmed by volume spike and filtered by weekly EMA200 trend for higher probability entries.
"""

name = "1d_ThreeBarReversal_LiquiditySweep"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for equal high/low detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Find equal highs/lows within 0.1% tolerance (liquidity pools)
    def find_equal_levels(arr, tolerance=0.001):
        equal_high = np.zeros(len(arr), dtype=bool)
        equal_low = np.zeros(len(arr), dtype=bool)
        for i in range(2, len(arr)):
            # Check if current high equals any of last 5 bars
            for j in range(max(0, i-5), i):
                if abs(arr[i] - arr[j]) / arr[j] < tolerance:
                    equal_high[i] = True
                    break
            # Check if current low equals any of last 5 bars
            for j in range(max(0, i-5), i):
                if abs(arr[i] - arr[j]) / arr[j] < tolerance:
                    equal_low[i] = True
                    break
        return equal_high, equal_low
    
    equal_high_1d, equal_low_1d = find_equal_levels(high_1d), find_equal_levels(low_1d)
    
    # Align equal high/low signals to lower timeframe
    equal_high_aligned = align_htf_to_ltf(prices, df_1d, equal_high_1d.astype(float))
    equal_low_aligned = align_htf_to_ltf(prices, df_1d, equal_low_1d.astype(float))
    
    # Three-bar reversal pattern detection
    # Bullish: down, down, up with higher close
    # Bearish: up, up, down with lower close
    bullish_reversal = np.zeros(n, dtype=bool)
    bearish_reversal = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        if (close[i-2] > close[i-1] and  # first bar down
            close[i-1] > close[i] and   # second bar down
            close[i] > close[i-1]):     # third bar up (reversal)
            bullish_reversal[i] = True
        if (close[i-2] < close[i-1] and  # first bar up
            close[i-1] < close[i] and   # second bar up
            close[i] < close[i-1]):     # third bar down (reversal)
            bearish_reversal[i] = True
    
    # Get weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike: current volume > 2x 20-day average (institutional participation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(equal_high_aligned[i]) or np.isnan(equal_low_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: liquidity sweep below (equal low) + bullish reversal + volume spike + above weekly EMA200
            if (equal_low_aligned[i] and bullish_reversal[i] and vol_spike and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: liquidity sweep above (equal high) + bearish reversal + volume spike + below weekly EMA200
            elif (equal_high_aligned[i] and bearish_reversal[i] and vol_spike and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below entry low or reversal signal
            entry_low = low[i-1] if i > 0 else low[i]
            if close[i] < entry_low or bearish_reversal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above entry high or reversal signal
            entry_high = high[i-1] if i > 0 else high[i]
            if close[i] > entry_high or bullish_reversal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals