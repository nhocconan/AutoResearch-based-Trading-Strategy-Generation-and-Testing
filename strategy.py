# Solution
#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation.
Long when price breaks above Donchian(20) high, 1-day EMA50 rising, and volume > 1.5x average.
Short when price breaks below Donchian(20) low, 1-day EMA50 falling, and volume > 1.5x average.
Exit when price crosses opposite Donchian boundary or 1-day EMA50 reverses direction.
Donchian provides clear breakout levels; 1-day EMA50 filters higher timeframe trend; volume confirms strength.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following daily trend while using 4h Donchian for entries.
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
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high, 1-day EMA50 rising, volume > 1.5x average
            if (close[i] > high_20[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low, 1-day EMA50 falling, volume > 1.5x average
            elif (close[i] < low_20[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian low OR 1-day EMA50 turns down
                if (close[i] < low_20[i] or 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian high OR 1-day EMA50 turns up
                if (close[i] > high_20[i] or 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_VolumeSpike_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0