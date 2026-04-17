#!/usr/bin/env python3
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
    
    # Get 12h data for trend filter and breakout levels
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA34 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema34_12h = close_series_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h rolling high (24 periods) and low (24 periods)
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    high_24_12h = high_12h_series.rolling(window=24, min_periods=24).max().values
    low_24_12h = low_12h_series.rolling(window=24, min_periods=24).min().values
    
    # Align 12h high/low to 4h timeframe
    high_24_12h_aligned = align_htf_to_ltf(prices, df_12h, high_24_12h)
    low_24_12h_aligned = align_htf_to_ltf(prices, df_12h, low_24_12h)
    
    # Volume confirmation (10-period MA on 4h)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(10, 24, 34)  # volume MA10, 12h high/low lookback, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma10[i]) or 
            np.isnan(high_24_12h_aligned[i]) or 
            np.isnan(low_24_12h_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        if position == 0:
            # Long: price > 12h high (24) + volume filter + 12h uptrend
            if close[i] > high_24_12h_aligned[i] and volume_filter and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < 12h low (24) + volume filter + 12h downtrend
            elif close[i] < low_24_12h_aligned[i] and volume_filter and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < 12h low (24) or 12h trend turns down
            if close[i] < low_24_12h_aligned[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > 12h high (24) or 12h trend turns up
            if close[i] > high_24_12h_aligned[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_HighLowBreakout_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0