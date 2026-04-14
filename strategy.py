#!/usr/bin/env python3
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
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index (14-period)
    atr1 = np.maximum.reduce([
        df_1d['high'].values - df_1d['low'].values,
        np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
        np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    ])
    atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr1 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid division by zero
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h EMA(20) for trend filter
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.3x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # 20 for EMA and volume avg
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: Chop > 61.8 (ranging) + price > EMA20 + volume confirmation
            if chop_aligned[i] > 61.8 and price > ema20[i] and vol > 1.3 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: Chop > 61.8 (ranging) + price < EMA20 + volume confirmation
            elif chop_aligned[i] > 61.8 and price < ema20[i] and vol > 1.3 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Chop < 38.2 (trending) or price < EMA20
            if chop_aligned[i] < 38.2 or price < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Chop < 38.2 (trending) or price > EMA20
            if chop_aligned[i] < 38.2 or price > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Chop_EMA20_Volume_Filter"
timeframe = "4h"
leverage = 1.0