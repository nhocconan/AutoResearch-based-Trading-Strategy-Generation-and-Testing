#!/usr/bin/env python3
"""
12h_rolling_breakout_1d_trend_volume_v1
Hypothesis: 12h rolling price breakout (high/low of last 20 periods) combined with 1d EMA trend filter and volume confirmation.
Long when price breaks above 20-period high with volume confirmation and price above 1d EMA50.
Short when price breaks below 20-period low with volume confirmation and price below 1d EMA50.
Designed for 15-30 trades/year on 12h timeframe with clear breakout logic that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rolling_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h rolling window for breakout levels (20 periods)
    lookback = 20
    high_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Breakout conditions
        breakout_high = close[i] > high_max[i]
        breakout_low = close[i] < low_min[i]
        
        # 1d trend filter
        above_1d_ema50 = close[i] > ema50_1d_aligned[i]
        below_1d_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below rolling low or trend turns bearish
            if breakout_low or below_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above rolling high or trend turns bullish
            if breakout_high or above_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above rolling high with volume confirmation and bullish trend
            if breakout_high and vol_confirmed and above_1d_ema50:
                position = 1
                signals[i] = 0.25
            # Short: breakout below rolling low with volume confirmation and bearish trend
            elif breakout_low and vol_confirmed and below_1d_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals