#!/usr/bin/env python3
"""
1h_Volume_Spice_4h_Trend_Filter_v1
Hypothesis: On 1h timeframe, enter long when price breaks above 20-period high with volume spike (>1.5x avg) and 4h trend is up (price > 4h EMA20); enter short when price breaks below 20-period low with volume spike and 4h trend is down (price < 4h EMA20). Exit on opposite break or trend reversal. Uses volume to confirm institutional interest and 4h EMA20 for trend filter to avoid counter-trend trades. Designed for 15-30 trades/year to minimize fee drag while capturing momentum in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period high/low for breakout
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_20 = high_series.rolling(window=20, min_periods=20).max().values
    lowest_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 4h EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need 20-period high/low and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        highest = highest_20[i]
        lowest = lowest_20[i]
        vol_spike = volume_spike[i]
        ema_4h_val = ema_4h_aligned[i]
        
        if position == 0:
            # Long: price > 20-period high with volume spike and above 4h EMA20
            if price > highest and vol_spike and price > ema_4h_val:
                signals[i] = 0.20
                position = 1
            # Short: price < 20-period low with volume spike and below 4h EMA20
            elif price < lowest and vol_spike and price < ema_4h_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price < 20-period low or below 4h EMA20 (trend change)
            if price < lowest or price < ema_4h_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price > 20-period high or above 4h EMA20 (trend change)
            if price > highest or price > ema_4h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Volume_Spice_4h_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0