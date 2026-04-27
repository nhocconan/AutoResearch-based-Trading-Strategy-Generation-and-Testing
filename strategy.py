#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeChop
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation, and chop regime filter.
Designed for 4h timeframe targeting 75-200 total trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn.
In trending regimes (price > EMA50 for longs, < EMA50 for shorts) and low chop (CHOP < 61.8),
Donchian breakouts with volume spike capture strong momentum continuations.
Exit on trend reversal (close crosses EMA50) or high chop (CHOP >= 61.8).
Works in bull/bear markets: trend filter adapts direction, chop filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and chop regime
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 12h Chopiness Index (CHOP) regime filter
    atr_12h = pd.Series(np.maximum.reduce([
        df_12h['high'] - df_12h['low'],
        np.abs(df_12h['high'] - df_12h['close'].shift(1)),
        np.abs(df_12h['low'] - df_12h['close'].shift(1))
    ])).rolling(window=14, min_periods=14).mean().values
    
    sum_true_range = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_true_range / (highest_high - lowest_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop, additional_delay_bars=0)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need Donchian(20), EMA50, CHOP, vol avg
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_val = ema_50_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with EMA alignment, low chop, and volume spike
            long_condition = (close_val > upper and 
                            close_val > ema_val and 
                            chop_val < 61.8 and 
                            vol_spike)
            short_condition = (close_val < lower and 
                             close_val < ema_val and 
                             chop_val < 61.8 and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below EMA50 (trend reversal) OR high chop (range regime)
            if close_val < ema_val or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA50 (trend reversal) OR high chop (range regime)
            if close_val > ema_val or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeChop"
timeframe = "4h"
leverage = 1.0