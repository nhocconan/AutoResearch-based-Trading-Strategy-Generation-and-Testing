#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Donchian channel breakouts on 12h timeframe capture medium-term trends. 
The 1d EMA200 ensures trades align with daily trend, reducing whipsaws. 
Volume > 1.5x 20-period average confirms institutional participation.
Target: 15-30 trades/year (~60-120 total over 4 years) to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume"
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
    
    # Donchian Channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume Confirmation (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # 1d EMA200 Trend Filter
    df_1d = get_htf_data(prices, '1d')
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA200
            if close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA200
            if close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high + price above 1d EMA200 + volume confirmation
            if (close[i] > high_roll[i-1] and 
                close[i] > ema_200_aligned[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low + price below 1d EMA200 + volume confirmation
            elif (close[i] < low_roll[i-1] and 
                  close[i] < ema_200_aligned[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals