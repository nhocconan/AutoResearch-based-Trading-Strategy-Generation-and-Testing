#!/usr/bin/env python3
# 12h_PriceChannel_Breakout_With_1D_Trend_Filter
# Hypothesis: Donchian channel breakout on 12h chart filtered by 1d EMA200 trend.
# In bull markets (price > 1d EMA200): long on upper Donchian breakout.
# In bear markets (price < 1d EMA200): short on lower Donchian breakout.
# Uses volume confirmation (>1.5x average volume) to avoid false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_PriceChannel_Breakout_With_1D_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Donchian channels on 12h data
    period = 20
    
    # Upper channel: highest high of last 20 periods
    highest_high = np.full_like(high, np.nan)
    for i in range(period - 1, len(high)):
        highest_high[i] = np.max(high[i - period + 1:i + 1])
    
    # Lower channel: lowest low of last 20 periods
    lowest_low = np.full_like(low, np.nan)
    for i in range(period - 1, len(low)):
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
    
    # Calculate average volume for confirmation
    vol_ma = np.full_like(volume, np.nan)
    for i in range(period - 1, len(volume)):
        vol_ma[i] = np.mean(volume[i - period + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Determine trend from 1d EMA200
            uptrend = close[i] > ema200_1d_aligned[i]
            downtrend = close[i] < ema200_1d_aligned[i]
            
            # Long: uptrend + price breaks above upper Donchian + volume confirmation
            if uptrend and close[i] > highest_high[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price breaks below lower Donchian + volume confirmation
            elif downtrend and close[i] < lowest_low[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian or trend reverses
            if close[i] < lowest_low[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian or trend reverses
            if close[i] > highest_high[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals