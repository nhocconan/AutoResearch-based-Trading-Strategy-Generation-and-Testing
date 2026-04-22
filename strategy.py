#!/usr/bin/env python3
"""
Hypothesis: 6-hour price action with 12-hour trend filter and volume confirmation.
Long when price breaks above 12h Donchian high with rising volume, short when breaks below with rising volume.
12h EMA20 determines trend direction to avoid counter-trend trades.
Designed for low trade frequency by requiring Donchian breakout + volume spike + trend alignment.
Works in bull markets by catching breakouts, in bear markets by avoiding false breakouts via trend filter.
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
    
    # Load 12-hour data for Donchian and EMA - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian channels (20 periods)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA20 for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe
    donchian_high = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_low = align_htf_to_ltf(prices, df_12h, low_20)
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume spike detector (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high, volume spike, and above 12h EMA20
            if (close[i] > donchian_high[i] and 
                vol_spike[i] and 
                close[i] > ema20_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian low, volume spike, and below 12h EMA20
            elif (close[i] < donchian_low[i] and 
                  vol_spike[i] and 
                  close[i] < ema20_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below 12h Donchian low OR below 12h EMA20
                if (close[i] < donchian_low[i] or 
                    close[i] < ema20_12h_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above 12h Donchian high OR above 12h EMA20
                if (close[i] > donchian_high[i] or 
                    close[i] > ema20_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_Breakout_12hEMA20_Volume"
timeframe = "6h"
leverage = 1.0