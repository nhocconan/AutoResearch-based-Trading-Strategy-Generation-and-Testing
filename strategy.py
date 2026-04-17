#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_Volume_TrendFilter
Hypothesis: Donchian channel breakouts capture trends while avoiding whipsaw in chop.
Long when price breaks above 20-period high + volume > 1.5x average + 1w close > 1w EMA34.
Short when price breaks below 20-period low + volume > 1.5x average + 1w close < 1w EMA34.
Exit on opposite signal. Position size: ±0.25. Uses 12h primary with 1w trend filter.
Designed to work in both bull (trend capture) and bear (avoids false signals via 1w filter).
"""

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
    
    # Calculate Donchian channel (20-period)
    def donchian_channels(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # Volume confirmation (10-period MA on 12h)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 10, 34)  # Donchian, volume MA10, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(volume_ma10[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        # Donchian breakout signals
        bullish_breakout = close[i] > upper[i-1]  # Break above previous period's high
        bearish_breakout = close[i] < lower[i-1]  # Break below previous period's low
        
        if position == 0:
            # Long: bullish breakout + volume filter + 1w uptrend
            if bullish_breakout and volume_filter and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + volume filter + 1w downtrend
            elif bearish_breakout and volume_filter and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout (price breaks below lower band)
            if bearish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout (price breaks above upper band)
            if bullish_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0