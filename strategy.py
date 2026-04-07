#!/usr/bin/env python3
"""
12h_volume_spike_breakout_1w_trend_v1
Hypothesis: Volume spikes on 12h timeframe combined with breakout from 1-week high/low and 1-week trend filter.
This strategy captures strong momentum moves while avoiding choppy markets by requiring:
1. Volume > 2x 20-period average (significant participation)
2. Price breaks above 20-period high (long) or below 20-period low (short)
3. 1-week EMA trend alignment (avoid counter-trend trades)
Works in both bull and bear markets by trading breakouts with volume confirmation.
Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volume_spike_breakout_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter and high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA for trend (20-period)
    ema_20 = df_1w['close'].ewm(span=20, adjust=False).mean()
    
    # Weekly high and low for breakout levels (20-period)
    high_20 = df_1w['high'].rolling(window=20, min_periods=20).max()
    low_20 = df_1w['low'].rolling(window=20, min_periods=20).min()
    
    # Align all weekly data to 12h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20.values)
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20.values)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20.values)
    
    # Volume confirmation: 20-period average on 12h (10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x average volume
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly low or trend turns bearish
            if close[i] < low_20_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above weekly high or trend turns bullish
            if close[i] > high_20_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly high with volume and bullish trend
            if (close[i] > high_20_aligned[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly low with volume and bearish trend
            elif (close[i] < low_20_aligned[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals