#!/usr/bin/env python3
"""
1d Daily Range Breakout with 1w Trend Filter and Volume Spike
Hypothesis: Daily range breakouts capture momentum with weekly trend filter.
Using 1w EMA200 as trend filter provides strong trend identification.
Volume spikes confirm institutional participation.
Designed for 1d timeframe with target 7-25 trades/year to minimize fee drag.
Works in both bull and bear regimes by following the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_daily_range_breakout_1w_trend_volume_v2"
timeframe = "1d"
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
    
    # Daily Range Breakout (using previous day's high/low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan  # First value invalid
    prev_low[0] = np.nan
    
    # Volume Spike Detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Strong volume spike
    
    # 1w EMA200 Trend Filter
    df_1w = get_htf_data(prices, '1w')
    ema_200 = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1w EMA200 (trend change)
            if close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 1w EMA200 (trend change)
            if close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above previous day's high + above 1w EMA200 + volume spike
            if (close[i] > prev_high[i] and 
                close[i] > ema_200_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below previous day's low + below 1w EMA200 + volume spike
            elif (close[i] < prev_low[i] and 
                  close[i] < ema_200_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals