#!/usr/bin/env python3
# 1d_Weekly_Momentum_Breakout_v1
# Hypothesis: Combines weekly momentum (price above weekly EMA20) with daily breakouts of Donchian(20) channels
# and volume confirmation. Uses weekly trend filter to avoid counter-trend trades, reducing false signals in choppy markets.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag while capturing strong trends.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

name = "1d_Weekly_Momentum_Breakout_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: price breaks above Donchian high + above weekly EMA20 + volume
            if (close[i] > highest_20[i] and 
                close[i] > ema_20_1w_aligned[i] and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below weekly EMA20 + volume
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_20_1w_aligned[i] and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or weekly trend turns bearish
            if (close[i] < lowest_20[i] or 
                close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or weekly trend turns bullish
            if (close[i] > highest_20[i] or 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals