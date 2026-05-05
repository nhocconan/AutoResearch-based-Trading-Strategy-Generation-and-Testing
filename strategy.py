#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND CHOP(14) > 61.8 (range regime)
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND CHOP(14) > 61.8 (range regime)
# Exit when price touches Donchian(20) midpoint OR choppiness regime shifts to trending (CHOP < 38.2)
# Uses 4h primary timeframe with 1d HTF for choppiness calculation to reduce whipsaw
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Works in both bull and bear markets by focusing on mean reversion in ranging conditions

name = "4h_Donchian20_Volume_ChopRegime_MeanReversion"
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
    
    # Get 1d data ONCE before loop for choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (HHV - LLV))
    # Using ATR(14) and highest high/lowest low over 14 periods
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    hh_14 = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll_14 = df_1d['low'].rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(np.sum(atr_14) / np.log10(14) / range_14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels on 4h data
    donchian_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_20_mid = (donchian_20_high + donchian_20_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_20_high[i]) or 
            np.isnan(donchian_20_low[i]) or 
            np.isnan(donchian_20_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND range regime (CHOP > 61.8)
            if (close[i] > donchian_20_high[i] and 
                volume_filter[i] and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND range regime (CHOP > 61.8)
            elif (close[i] < donchian_20_low[i] and 
                  volume_filter[i] and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian midpoint OR regime shifts to trending (CHOP < 38.2)
            if (close[i] <= donchian_20_mid[i] or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Donchian midpoint OR regime shifts to trending (CHOP < 38.2)
            if (close[i] >= donchian_20_mid[i] or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals