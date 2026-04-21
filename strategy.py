#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter (bullish if price > weekly pivot, bearish if < weekly pivot) and volume confirmation (>1.5x 20-period MA). 
Weekly pivot provides structural bias from higher timeframe, reducing false breakouts in choppy markets. Designed for low trade frequency (~15-25/year) to minimize fee drag while capturing strong directional moves aligned with weekly structure.
Works in both bull and bear markets by using weekly pivot as trend filter: only long when above weekly pivot, only short when below.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Donchian calculation, 1w for weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d OHLC for Donchian(20) calculation (based on previous 20 daily bars) ===
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # Calculate Donchian channels: 20-period high and low
    high_20 = pd.Series(df_1d_high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d_low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # === 1w OHLC for weekly pivot calculation (based on previous weekly bar) ===
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_pivot = (df_1w_high + df_1w_low + df_1w_close) / 3.0
    
    # Align 1w weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike filter (1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) 
            or np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        upper_channel = high_20_aligned[i]
        lower_channel = low_20_aligned[i]
        weekly_pivot = weekly_pivot_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Enter long: price breaks above upper Donchian channel, above weekly pivot, with volume spike
            long_condition = (price > upper_channel) and (price > weekly_pivot) and volume_spike
            # Enter short: price breaks below lower Donchian channel, below weekly pivot, with volume spike
            short_condition = (price < lower_channel) and (price < weekly_pivot) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0