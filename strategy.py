#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d Trend + Volume Confirmation
Hypothesis: 4-hour Donchian channel (20-period) breakouts aligned with daily EMA trend
and volume confirmation capture strong momentum moves. Works in bull markets via
breakouts and in bear markets via breakdowns. Volume filters prevent false breaks.
Target: 20-50 trades per year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(21) for trend filter
    ema_21 = df_1d['close'].ewm(span=21, adjust=False).mean().values
    ema_21_4h = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # 4h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average on 4h)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_21_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] <= donchian_low[i] or close[i] < ema_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] >= donchian_high[i] or close[i] > ema_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above Donchian high with trend alignment
            if (close[i] > donchian_high[i] and 
                close[i] > ema_21_4h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakdown short below Donchian low with trend alignment
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_21_4h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals