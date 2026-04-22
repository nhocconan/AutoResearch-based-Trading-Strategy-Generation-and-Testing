#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian channel breakout with 1-day volatility filter and volume confirmation.
The Donchian channel (20-period high/low) identifies volatility breakouts. The 1-day ATR-based
volatility filter ensures we only trade when volatility is expanding, avoiding choppy markets.
Volume spikes confirm institutional participation. This combination works in both bull and bear
markets by capturing breakouts during volatility expansion periods. Target: 20-40 trades/year per symbol.
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
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period SMA of ATR to identify expanding volatility
    atr_ma_20 = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily ATR and its MA to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # Calculate 4h Donchian channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_20_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: current ATR > 20-period ATR mean (expanding volatility)
        vol_expanding = atr_14_aligned[i] > atr_ma_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, volatility expanding, volume spike
            if (close[i] > highest_20[i] and    # Break above Donchian high
                vol_expanding and               # Volatility expanding
                volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, volatility expanding, volume spike
            elif (close[i] < lowest_20[i] and   # Break below Donchian low
                  vol_expanding and             # Volatility expanding
                  volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low
                if close[i] < lowest_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian high
                if close[i] > highest_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolATR_Volume"
timeframe = "4h"
leverage = 1.0