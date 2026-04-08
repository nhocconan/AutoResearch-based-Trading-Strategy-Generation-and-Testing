#!/usr/bin/env python3
"""
12h Donchian Breakout + Daily EMA + Volume Confirmation
Hypothesis: Donchian channel breakouts (20-period) from daily timeframe capture strong momentum.
Breakouts above upper band or below lower band with daily EMA trend alignment and volume confirmation.
Designed for 12h timeframe to balance trade frequency and signal quality in both bull and bear markets.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_ema_volume_v1"
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
    
    # Daily data for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period) from previous day
    # Using previous day's values to avoid look-ahead
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Donchian upper/lower bands (20-period high/low)
    donch_high = prev_high.rolling(window=20, min_periods=20).max()
    donch_low = prev_low.rolling(window=20, min_periods=20).min()
    
    # Align to 12h timeframe (previous day's levels are known at open)
    donch_high_12h = align_htf_to_ltf(prices, df_1d, donch_high.values)
    donch_low_12h = align_htf_to_ltf(prices, df_1d, donch_low.values)
    
    # Daily EMA(21) for trend filter
    ema_21 = df_1d['close'].ewm(span=21, adjust=False).mean().values
    ema_21_12h = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Volume filter (>1.5x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or 
            np.isnan(ema_21_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian band or trend reverses
            if close[i] < donch_low_12h[i] or close[i] < ema_21_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian band or trend reverses
            if close[i] > donch_high_12h[i] or close[i] > ema_21_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at upper Donchian with trend alignment
            if (close[i] >= donch_high_12h[i] and 
                close[i] > ema_21_12h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at lower Donchian with trend alignment
            elif (close[i] <= donch_low_12h[i] and 
                  close[i] < ema_21_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals