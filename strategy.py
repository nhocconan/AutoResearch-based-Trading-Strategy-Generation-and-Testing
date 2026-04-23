#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) Breakout with 1d EMA34 Trend Filter and Volume Spike
- Uses 12h Donchian channel breakout for entry signals
- 1d EMA34 defines higher timeframe trend filter: only trade in direction of daily trend
- Volume confirmation (> 2.0x 30-period average) filters weak signals
- Exit on opposite Donchian breakout or trend reversal
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by following higher timeframe trend on breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian(20) - upper and lower bands
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 30, 20)  # for EMA34, volume MA, and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price above 1d EMA34 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower AND price below 1d EMA34 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: opposite Donchian breakout OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long when price breaks below Donchian lower OR price closes below 1d EMA34
                if (close[i] < lowest_low[i] or close[i] < ema_34_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price breaks above Donchian upper OR price closes above 1d EMA34
                if (close[i] > highest_high[i] or close[i] > ema_34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0