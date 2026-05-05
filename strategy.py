#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when: price breaks above 20-period 4h Donchian high, 1d ATR(14) > 1.5x its 50-period MA, and volume > 1.5x 20-period 4h volume MA
# Short when: price breaks below 20-period 4h Donchian low, 1d ATR(14) > 1.5x its 50-period MA, and volume > 1.5x 20-period 4h volume MA
# Exit when price returns to the opposite Donchian level (mean reversion in choppy markets) or ATR filter fails
# Uses Donchian channels for structure (works in trending markets) and ATR filter to avoid low-volatility whipsaws
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for ATR(14) and its MA(50)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    if len(high_1d) >= 15:
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    else:
        atr_14 = np.full(len(close_1d), np.nan)
    
    # Calculate 50-period MA of ATR(14) for volatility regime filter
    if len(atr_14) >= 50:
        atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
        atr_filter = atr_14 > (1.5 * atr_ma_50)  # High volatility regime
    else:
        atr_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Align ATR filter to 4h timeframe
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Calculate 4h Donchian(20) channels
    if len(high) >= 20 and len(low) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(atr_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, high volatility, volume confirmation
            if (close[i] > donch_high[i] and 
                open_price[i] <= donch_high[i] and  # Breakout confirmation on this bar
                atr_filter_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, high volatility, volume confirmation
            elif (close[i] < donch_low[i] and 
                  open_price[i] >= donch_low[i] and  # Breakdown confirmation on this bar
                  atr_filter_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low (mean reversion) or volatility filter fails
            if close[i] < donch_low[i] or not atr_filter_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high (mean reversion) or volatility filter fails
            if close[i] > donch_high[i] or not atr_filter_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals