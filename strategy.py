#!/usr/bin/env python3
"""
1d Donchian Breakout + 1w EMA Trend + Volume + ATR Stop
Hypothesis: Daily Donchian breakouts capture sustained trends. Filtered by weekly EMA trend (more stable) and volume confirmation. ATR-based stop manages risk. Works in bull/bear by using volatility-adjusted stops and trend alignment. Targets 7-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = df_1w['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 1d ATR(14) for stop loss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend reverses OR ATR stop
            if (close[i] <= lowest_low[i] or 
                close[i] < ema_20_1w_aligned[i] or
                close[i] <= (highest_high[i-1] - 2.5 * atr[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend reverses OR ATR stop
            if (close[i] >= highest_high[i] or 
                close[i] > ema_20_1w_aligned[i] or
                close[i] >= (lowest_low[i-1] + 2.5 * atr[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] > highest_high[i-1] and 
                close[i] > ema_20_1w_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with trend alignment and volume
            elif (close[i] < lowest_low[i-1] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals