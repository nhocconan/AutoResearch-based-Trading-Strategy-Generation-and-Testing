#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA34 trend filter, volume confirmation, and choppiness regime filter.
Designed for 4h timeframe targeting 75-200 total trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn.
In trending regimes (price > EMA34 for longs, < EMA34 for shorts), breakouts at R1/S1 with volume spike capture momentum.
Choppiness filter avoids whipsaws in ranging markets. Works in both bull and bear markets by following the 1d trend.
Exit on trend reversal (close crosses EMA34).
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
    
    # Get 1d data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous 1d bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    # Avoid division by zero in case of flat day
    rng = np.where(rng == 0, 1e-10, rng)
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter (using 1d data)
    # Higher values = more choppy, lower = more trending
    # We use choppy > 61.8 as ranging (avoid entries), < 38.2 as trending (allow entries)
    atr_1d = pd.Series(np.maximum.reduce([
        df_1d['high'] - df_1d['low'],
        np.abs(df_1d['high'] - df_1d['close'].shift(1)),
        np.abs(df_1d['low'] - df_1d['close'].shift(1))
    ])).rolling(window=14, min_periods=14).sum().values
    
    true_range_sum = atr_1d
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10((max_high - min_low) * 14) / np.log10(2)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(true_range_sum / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 1d shift, EMA34, vol avg, chop calculation
    start_idx = max(30, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with EMA alignment, volume spike, and trending regime (low chop)
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            chop_val < 38.2)  # Trending regime
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             chop_val < 38.2)  # Trending regime
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below EMA34 (trend reversal) OR chop becomes too high (ranging market)
            if close_val < ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA34 (trend reversal) OR chop becomes too high (ranging market)
            if close_val > ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0