#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + Choppiness Regime Filter
Long when price breaks above Donchian(20) high with volume spike in ranging market (chop > 61.8)
Short when price breaks below Donchian(20) low with volume spike in ranging market
Exit when price crosses Donchian midpoint
Designed to capture breakouts in ranging markets with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # === Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # === Choppiness Index (14) ===
    atr = pd.Series(np.sqrt((high - low)**2)).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Fix: Calculate properly using rolling sum
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high_14 - lowest_low_14
    chop = 100 * np.log10(atr_sum / hh_ll_diff) / np.log10(14)
    chop = np.where(hh_ll_diff > 0, chop, 50)  # Avoid division by zero
    
    # === 1d Trend Filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(chop[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout conditions with volume spike and chop filter (ranging market)
            if close[i] > highest_high[i] and volume_spike[i] and chop[i] > 61.8:
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and volume_spike[i] and chop[i] > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals