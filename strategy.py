#!/usr/bin/env python3
"""
12h_donchian_1w_trend_volume_v1
Hypothesis: Donchian breakout on 12h with weekly EMA trend filter and volume confirmation works in both bull and bear markets.
Weekly EMA determines trend direction, Donchian(20) provides entry signals, volume confirms breakouts.
Target: 12-37 trades/year (50-150 over 4 years) with low turnover to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channels on 12h (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average on 12h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR EMA turns down
            if close[i] < donchian_low[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR EMA turns up
            if close[i] > donchian_high[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout in uptrend
            if (close[i] >= donchian_high[i] and 
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short breakout in downtrend
            elif (close[i] <= donchian_low[i] and 
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals