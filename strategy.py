#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d Trend + Volume Confirmation v2
Hypothesis: Donchian channel breakouts capture momentum in both bull and bear markets.
Trend filtered by daily EMA(21) ensures directional alignment. Volume > 1.5x average
confirms institutional participation. Stops when price closes below/above entry bar's
extreme. Tightened entry conditions to reduce trade frequency and improve robustness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
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
    
    # Donchian channel (25-period) - longer period for fewer, higher-quality signals
    donchian_len = 25
    high_max = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    low_min = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Shift by 1 to use only completed periods
    upper = np.roll(high_max, 1)
    lower = np.roll(low_min, 1)
    upper[0] = np.nan
    lower[0] = np.nan
    
    # 1d EMA(21) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_21 = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Volume filter (>1.8x 25-period average for stricter confirmation)
    vol_ma = pd.Series(volume).rolling(window=donchian_len, min_periods=donchian_len).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_len, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below entry bar's low or trend reverses
            if close[i] < lower[i] or close[i] < ema_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above entry bar's high or trend reverses
            if close[i] > upper[i] or close[i] > ema_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price closes above upper band with volume and trend alignment
            if (close[i] > upper[i] and 
                close[i] > ema_21_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakout: price closes below lower band with volume and trend alignment
            elif (close[i] < lower[i] and 
                  close[i] < ema_21_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals