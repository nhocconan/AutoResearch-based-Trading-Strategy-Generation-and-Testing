#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR for Choppiness Index
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean()
    atr_sum_14 = atr_1d.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low_14 = df_1d['low'].rolling(window=14, min_periods=14).min()
    
    # Choppiness Index formula
    chop = 100 * np.log10(atr_sum_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_values = chop.values
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 1d volume confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 4h Donchian breakout levels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for volatility filter
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_aligned[i]
        
        # Price levels
        price = close[i]
        upper_donchian = highest_high_20[i]
        lower_donchian = lowest_low_20[i]
        
        # Chop regime: chop > 61.8 = range, chop < 38.2 = trending
        chop_value = chop_aligned[i]
        in_range = chop_value > 61.8
        in_trend = chop_value < 38.2
        
        # Volatility filter: only trade when ATR > 15-period average
        atr_avg_15 = pd.Series(atr_4h).rolling(window=15, min_periods=15).mean()[i]
        vol_filter = atr_4h[i] > atr_avg_15
        
        # Entry conditions: Donchian breakout in correct regime with volume
        long_signal = vol_confirm and vol_filter and in_trend and (price > upper_donchian)
        short_signal = vol_confirm and vol_filter and in_trend and (price < lower_donchian)
        
        # Exit conditions: opposite Donchian breakout
        long_exit = price < lower_donchian
        short_exit = price > upper_donchian
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals