#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: Donchian(20) breakout on 12h with 1d trend filter and volume confirmation.
Long when price breaks above 12h Donchian upper channel AND price above 1d EMA200 (uptrend).
Short when price breaks below 12h Donchian lower channel AND price below 1d EMA200 (downtrend).
Volume confirmation filters weak signals. Exit on Donchian middle line or trend reversal.
Designed for low trade frequency (<30/year) to minimize fee drag in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    ema_200 = df_1d['close'].ewm(span=200, adjust=False).mean()
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200.values)
    
    # 12h Donchian channels (20-period)
    # Calculate on 12h data directly
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0  # middle line for exit
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price below Donchian middle OR trend turns down
            if close[i] < mid_20[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price above Donchian middle OR trend turns up
            if close[i] > mid_20[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above upper Donchian with volume and uptrend
            if (close[i] > high_20[i] and vol_confirm and 
                close[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: break below lower Donchian with volume and downtrend
            elif (close[i] < low_20[i] and vol_confirm and 
                  close[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals