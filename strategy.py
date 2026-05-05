#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND chop(14) < 38.2 (trending regime)
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND chop(14) < 38.2 (trending regime)
# Exit when price touches Donchian(20) midpoint OR volume drops below average
# Uses 4h primary timeframe to balance trade frequency and signal quality
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Donchian channels provide clear structure, volume confirms participation, chop filter avoids ranging markets

name = "4h_Donchian20_Breakout_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Choppiness Index (14) - values < 38.2 indicate trending regime
    if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
        # True Range
        tr1 = pd.Series(high).rolling(window=14, min_periods=14).max().values - pd.Series(low).rolling(window=14, min_periods=14).min().values
        tr2 = np.abs(pd.Series(high).shift(1).values - pd.Series(close).shift(1).values)
        tr3 = np.abs(pd.Series(low).shift(1).values - pd.Series(close).shift(1).values)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        # Choppiness Index = 100 * log10(sum(ATR14) / (ATR14 * 14)) / log10(14)
        sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
        chop = 100 * np.log10(sum_atr_14 / (atr_14 * 14)) / np.log10(14)
        chop_filter = chop < 38.2  # Trending regime
    else:
        chop = np.full(n, np.nan)
        chop_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high + volume + trending regime
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + volume + trending regime
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian midpoint OR volume drops below average
            if close[i] <= donchian_mid[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Donchian midpoint OR volume drops below average
            if close[i] >= donchian_mid[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals