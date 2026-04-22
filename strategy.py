#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Long when price breaks above Donchian(20) high, 1-day EMA34 rising, and volume spikes above 1.5x average.
Short when price breaks below Donchian(20) low, 1-day EMA34 falling, and volume spikes above 1.5x average.
Exit when price returns to Donchian midpoint or trend reverses.
Designed for low trade frequency by requiring multiple confirmations and using 12h timeframe.
Works in both bull and bear markets by following daily trend while using 12h Donchian for entries.
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
    
    # Load 1-day data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_20
    donchian_low = low_20
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, EMA34 rising, volume spike
            if (close[i] > donchian_high[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, EMA34 falling, volume spike
            elif (close[i] < donchian_low[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls to Donchian mid OR EMA34 starts falling
                if (close[i] <= donchian_mid[i] or 
                    ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises to Donchian mid OR EMA34 starts rising
                if (close[i] >= donchian_mid[i] or 
                    ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0