#!/usr/bin/env python3
"""
12h Donchian Breakout with 1w Trend Filter and Volume Confirmation
Hypothesis: 12h timeframe balances trade frequency and signal quality. 
Breakouts from 20-period Donchian channels capture strong momentum moves.
1-week EMA50 filter ensures trades align with higher-timeframe trend, reducing false signals.
Volume confirmation (>1.5x 20-period average) adds conviction. 
Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA50 for trend filter (Higher Timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian Channel (20-period)
    # Highest high and lowest low over past 20 bars (including current)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period volume average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    # Start after warmup period
    start_idx = 50  # Ensures all indicators are valid
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        
        if position == 0:
            # Long breakout: price breaks above 20-period high, above 1w EMA50, with volume confirmation
            if (price > highest_high[i] and 
                price > ema_50_1w_aligned[i] and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 20-period low, below 1w EMA50, with volume confirmation
            elif (price < lowest_low[i] and 
                  price < ema_50_1w_aligned[i] and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 20-period low (stop and reverse condition)
            if price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 20-period high (stop and reverse condition)
            if price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0