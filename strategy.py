#!/usr/bin/env python3
"""
4h_Structure_Volume_V1
Strategy: 4h Donchian(20) breakout with 12h EMA200 trend filter and volume confirmation.
Long: Price breaks above Donchian high AND price above 12h EMA200 AND volume > 1.5x 20-bar avg.
Short: Price breaks below Donchian low AND price below 12h EMA200 AND volume > 1.5x 20-bar avg.
Exit: Price re-enters Donchian channel OR volume drops below average.
Position size: 0.25 (25% of capital).
Target: ~100 total trades over 4 years (~25/year).
Works in bull (breakouts) and bear (breakdowns) with volume confirmation to avoid false signals.
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
    
    # === Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation: 1.5x 20-bar average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # === 12h EMA200 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_200_12h = pd.Series(df_12h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_threshold[i]) or 
            np.isnan(ema_200_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian high + above 12h EMA200 + volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema_200_12h_aligned[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + below 12h EMA200 + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_200_12h_aligned[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volume drops below average
            if (close[i] < donchian_high[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volume drops below average
            if (close[i] > donchian_low[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Structure_Volume_V1"
timeframe = "4h"
leverage = 1.0