#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: Donchian(20) breakout on 4h with 1d EMA50 trend filter and volume spike confirmation. 
Enter long when price breaks above upper Donchian channel in uptrend with volume > 1.5x average.
Enter short when price breaks below lower Donchian channel in downtrend with volume > 1.5x average.
Uses ATR-based stoploss (2*ATR) and discrete sizing 0.25 to limit trades (~20-40/year).
Works in bull/bear via 1d trend filter and volatility-adjusted breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR for volatility filtering and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate volume average (20-period) for volume spike confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 20 for Donchian/volume, 14 for ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_val = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr[i]
        vol_average = vol_avg[i]
        size = fixed_size
        
        # Volume spike condition: current volume > 1.5 x average volume
        volume_spike = vol_val > 1.5 * vol_average
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above upper Donchian in uptrend with volume spike
            long_entry = (close_val > upper_channel) and (close_val > ema_50_val) and volume_spike
            # Short: price breaks below lower Donchian in downtrend with volume spike
            short_entry = (close_val < lower_channel) and (close_val < ema_50_val) and volume_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or ATR-based stop
            # Exit if price breaks below lower Donchian OR drops 2*ATR from entry
            if close_val < lower_channel or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or ATR-based stop
            # Exit if price breaks above upper Donchian OR rises 2*ATR from entry
            if close_val > upper_channel or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0