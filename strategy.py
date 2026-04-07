#!/usr/bin/env python3
"""
6h_donchian_1w_trend_volume_v1
Hypothesis: Donchian breakout on 6h with weekly trend filter and volume confirmation.
In trending markets (price above/below weekly 20-period EMA), breakouts in trend direction.
In ranging markets, fade at Donchian bands with volume exhaustion.
Volume confirmation reduces false signals. Targets 15-25 trades/year (60-100 over 4 years).
Weekly trend filter adapts to bull/bear markets via higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_1w_trend_volume_v1"
timeframe = "6h"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema20_6h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 60-period Donchian channels on 6h (20 periods * 3 for 6h to approximate daily)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 20-period volume average on 6h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema20_6h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR 
            # price exceeds Donchian high and weekly trend turns down
            if close[i] < lowest_low[i] or (close[i] > highest_high[i] and close[i] < ema20_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR
            # price goes below Donchian low and weekly trend turns up
            if close[i] > highest_high[i] or (close[i] < lowest_low[i] and close[i] > ema20_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout longs in uptrend (price > weekly EMA)
            if (close[i] >= highest_high[i] and 
                vol_confirm and 
                close[i] > ema20_6h[i]):
                position = 1
                signals[i] = 0.25
            # Breakout shorts in downtrend (price < weekly EMA)
            elif (close[i] <= lowest_low[i] and 
                  vol_confirm and 
                  close[i] < ema20_6h[i]):
                position = -1
                signals[i] = -0.25
            # Mean reversion longs at Donchian low in ranging market (price near weekly EMA)
            elif (close[i] <= lowest_low[i] and 
                  vol_confirm and 
                  abs(close[i] - ema20_6h[i]) < (highest_high[i] - lowest_low[i]) * 0.3):
                position = 1
                signals[i] = 0.20
            # Mean reversion shorts at Donchian high in ranging market
            elif (close[i] >= highest_high[i] and 
                  vol_confirm and 
                  abs(close[i] - ema20_6h[i]) < (highest_high[i] - lowest_low[i]) * 0.3):
                position = -1
                signals[i] = -0.20
    
    return signals