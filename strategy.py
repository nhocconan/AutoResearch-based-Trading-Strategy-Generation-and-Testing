#!/usr/bin/env python3
"""
4h_donchian_12h_trend_volume_v2
Hypothesis: 4h Donchian(20) breakout in direction of 12h EMA20 trend with volume confirmation.
In bull markets: buy breakouts above upper band when price > EMA20.
In bear markets: sell breakdowns below lower band when price < EMA20.
Volume filter reduces false breakouts. Targets 20-50 trades/year (80-200 over 4 years).
Works in both bull and bear markets by following 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_trend_volume_v2"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema20_4h = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average on 4h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema20_4h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below midline (mean reversion) OR trend changes
            midline = (high_max[i] + low_min[i]) / 2
            if close[i] < midline or (close[i] < ema20_4h[i] and ema20_4h[i] < ema20_4h[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above midline OR trend changes
            midline = (high_max[i] + low_min[i]) / 2
            if close[i] > midline or (close[i] > ema20_4h[i] and ema20_4h[i] > ema20_4h[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above upper band in uptrend
            if (close[i] > high_max[i] and 
                vol_confirm and 
                close[i] > ema20_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below lower band in downtrend
            elif (close[i] < low_min[i] and 
                  vol_confirm and 
                  close[i] < ema20_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals