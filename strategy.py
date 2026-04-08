#!/usr/bin/env python3
"""
12h Camarilla Pivot + 1w Trend + Volume Confirmation
Hypothesis: Camarilla pivot levels act as strong support/resistance. Trade bounces off these levels with weekly trend filter and volume confirmation. Works in bull/bear by using mean-reversion at extremes with trend alignment. Targets 15-30 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = df_1w['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Camarilla pivots (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels from previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3, H4, L4
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H4 = close + 1.5*(high-low)/2, L4 = close - 1.5*(high-low)/2
    range_1d = high_1d - low_1d
    h3 = close_1d + 1.1 * range_1d / 4
    l3 = close_1d - 1.1 * range_1d / 4
    h4 = close_1d + 1.5 * range_1d / 2
    l4 = close_1d - 1.5 * range_1d / 2
    
    # Align to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume filter (>1.3x 50-period average)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 OR trend reverses
            if (close[i] < l3_aligned[i] or close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 OR trend reverses
            if (close[i] > h3_aligned[i] or close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches L3/L4 with weekly uptrend and volume
            if (close[i] <= l3_aligned[i] and 
                close[i] > l4_aligned[i] and  # Avoid breaking too far
                close[i] > ema_50_1w_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H3/H4 with weekly downtrend and volume
            elif (close[i] >= h3_aligned[i] and 
                  close[i] < h4_aligned[i] and  # Avoid breaking too far
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals