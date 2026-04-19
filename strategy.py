#!/usr/bin/env python3
"""
12h_WickReversal_Trend_Filter
Hypothesis: On 12h timeframe, long wicks indicate rejection of higher/lower prices and potential reversals.
Combined with 1d trend filter (EMA34) and volume confirmation to avoid false signals.
Designed to work in both bull and bear markets by focusing on rejection signals with trend alignment.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

name = "12h_WickReversal_Trend_Filter"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # Calculate upper and lower wick ratios
    body_size = np.abs(close - open_prices) if 'open' in prices else np.abs(close - np.roll(close, 1))
    upper_wick = high - np.maximum(close, open_prices) if 'open' in prices else high - np.maximum(close, np.roll(close, 1))
    lower_wick = np.minimum(close, open_prices) - low if 'open' in prices else np.minimum(close, np.roll(close, 1)) - low
    
    # Handle case where open column might not exist
    if 'open' not in prices:
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]
        body_size = np.abs(close - open_prices)
        upper_wick = high - np.maximum(close, open_prices)
        lower_wick = np.minimum(close, open_prices) - low
    
    # Avoid division by zero
    body_size_safe = np.where(body_size == 0, 1, body_size)
    upper_wick_ratio = upper_wick / body_size_safe
    lower_wick_ratio = lower_wick / body_size_safe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(upper_wick_ratio[i]) or np.isnan(lower_wick_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: long lower wick (rejection of lower prices) in uptrend
            if (lower_wick_ratio[i] > 2.0 and  # Significant lower wick
                close[i] > ema34_1d_aligned[i] and  # Price above 1d EMA34 (uptrend)
                volume_confirm[i]):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: long upper wick (rejection of higher prices) in downtrend
            elif (upper_wick_ratio[i] > 2.0 and  # Significant upper wick
                  close[i] < ema34_1d_aligned[i] and  # Price below 1d EMA34 (downtrend)
                  volume_confirm[i]):  # Volume confirmation
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below EMA34 or wick signal reverses
            if (close[i] < ema34_1d_aligned[i]) or (upper_wick_ratio[i] > 2.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above EMA34 or wick signal reverses
            if (close[i] > ema34_1d_aligned[i]) or (lower_wick_ratio[i] > 2.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals