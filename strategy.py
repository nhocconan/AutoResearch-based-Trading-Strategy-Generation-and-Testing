# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_donchian_1w_trend_volume_v1
Hypothesis: Donchian channel breakouts on 4h, filtered by 1-week trend (EMA200) and volume confirmation, work in both bull and bear markets.
In bull markets, buy breakouts above upper band; in bear markets, sell breakdowns below lower band.
Volume confirmation reduces false signals. Targets 20-50 trades/year (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1w_trend_volume_v1"
timeframe = "4h"
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
    
    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_4h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Donchian channel (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average on 4h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema200_4h[i]) or 
            np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band
            if close[i] < low_min[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band
            if close[i] > high_max[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above upper band in uptrend (close > EMA200)
            if (close[i] >= high_max[i] and 
                vol_confirm and 
                close[i] > ema200_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakdown below lower band in downtrend (close < EMA200)
            elif (close[i] <= low_min[i] and 
                  vol_confirm and 
                  close[i] < ema200_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals