#!/usr/bin/env python3
"""
1h_moving_average_crossover_volume_4h1d_trend_filter
Hypothesis: On 1h timeframe, use EMA crossover (12/26) with volume confirmation and 4h/1d trend filters. Enter long when EMA12 crosses above EMA26 with volume > 1.5x average, EMA12 > EMA26 on 4h, and price > 200-period SMA on 1d. Enter short when EMA12 crosses below EMA26 with volume > 1.5x average, EMA12 < EMA26 on 4h, and price < 200-period SMA on 1d. Exit on opposite crossover. This strategy captures medium-term momentum with trend alignment from higher timeframes, reducing false signals in choppy markets. Targets 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_moving_average_crossover_volume_4h1d_trend_filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA crossover on 1h
    ema12 = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # 4h EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 26:
        return np.zeros(n)
    ema12_4h = pd.Series(df_4h['close'].values).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26_4h = pd.Series(df_4h['close'].values).ewm(span=26, min_periods=26, adjust=False).mean().values
    ema12_4h_aligned = align_htf_to_ltf(prices, df_4h, ema12_4h)
    ema26_4h_aligned = align_htf_to_ltf(prices, df_4h, ema26_4h)
    
    # 1d trend filter (price vs 200-period SMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    sma200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # Volume confirmation (24-period average on 1h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema12[i]) or np.isnan(ema26[i]) or 
            np.isnan(ema12_4h_aligned[i]) or np.isnan(ema26_4h_aligned[i]) or
            np.isnan(sma200_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if EMA12 crosses below EMA26
            if ema12[i] < ema26[i] and ema12[i-1] >= ema26[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit if EMA12 crosses above EMA26
            if ema12[i] > ema26[i] and ema12[i-1] <= ema26[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: EMA12 crosses above EMA26 with volume confirmation and trend filters
            long_entry = False
            if (ema12[i] > ema26[i] and ema12[i-1] <= ema26[i-1] and
                vol_confirm and
                ema12_4h_aligned[i] > ema26_4h_aligned[i] and
                close[i] > sma200_1d_aligned[i]):
                long_entry = True
            
            # Short entry: EMA12 crosses below EMA26 with volume confirmation and trend filters
            short_entry = False
            if (ema12[i] < ema26[i] and ema12[i-1] >= ema26[i-1] and
                vol_confirm and
                ema12_4h_aligned[i] < ema26_4h_aligned[i] and
                close[i] < sma200_1d_aligned[i]):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
    
    return signals