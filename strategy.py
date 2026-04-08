#!/usr/bin/env python3
"""
4H Donchian Breakout + 12H Trend + Volume Confirmation
Hypothesis: Donchian channel breakouts on 4h with 12h EMA trend filter and volume confirmation capture strong momentum. 
The 12h EMA ensures alignment with higher timeframe trend, reducing whipsaws. Volume confirms breakout strength.
Designed for 4h timeframe to achieve 20-50 trades/year, balancing signal quality and frequency.
Works in bull markets via breakouts and in bear via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend filter
    ema_12h = df_12h['close'].ewm(span=34, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] <= donchian_low[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] >= donchian_high[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long: price closes above Donchian high with trend and volume
            if (close[i] > donchian_high[i] and 
                close[i] > ema_12h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short: price closes below Donchian low with trend and volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_12h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals