#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR(14) volatility filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, 1d ATR(14) > 1.5x its 20-period average (high volatility), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, 1d ATR(14) > 1.5x its 20-period average, volume > 1.5x avg
# Exit when: price retraces to midpoint of Donchian channel OR opposite breakout occurs
# Uses daily volatility filter to avoid low-volatility chop, targeting 75-200 total trades over 4 years
# High volatility breakouts capture momentum in both bull and bear markets, reducing false breakouts

name = "12h_donchian20_1datr_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 12h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_threshold = 1.5 * atr_ma
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_threshold_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volatility filter + volume
            if (volume[i] > volume_threshold[i] and 
                atr_14_aligned[i] > atr_threshold_aligned[i]):
                if close[i] > high_20[i]:
                    # Bullish breakout above Donchian high with high volatility
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i]:
                    # Bearish breakdown below Donchian low with high volatility
                    signals[i] = -0.25
                    position = -1
    
    return signals