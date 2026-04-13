#!/usr/bin/env python3
"""
6h_1w_pullback_reversion
Hypothesis: In strong weekly uptrends (price > weekly 200-EMA), buy pullbacks to the 6-hour 20-EMA with volume confirmation. In strong weekly downtrends (price < weekly 200-EMA), sell rallies to the 6-hour 20-EMA with volume confirmation. Uses weekly trend filter to avoid counter-trend trades, targeting 15-30 trades/year. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly 200 EMA for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_uptrend = close_1w > ema200_1w
    weekly_downtrend = close_1w < ema200_1w
    
    # 6-hour 20 EMA for entry
    close_series = pd.Series(close)
    ema20_6h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6-hour volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma_20 * 1.3)
    
    # Align weekly signals to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(ema20_6h[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: weekly uptrend + price at 6h 20-EMA + volume
        long_entry = (weekly_uptrend_aligned[i] > 0.5 and 
                     close[i] >= ema20_6h[i] * 0.998 and  # Allow small tolerance
                     close[i] <= ema20_6h[i] * 1.002 and
                     volume_confirm[i] > 0.5)
        
        # Short entry: weekly downtrend + price at 6h 20-EMA + volume
        short_entry = (weekly_downtrend_aligned[i] > 0.5 and 
                      close[i] >= ema20_6h[i] * 0.998 and
                      close[i] <= ema20_6h[i] * 1.002 and
                      volume_confirm[i] > 0.5)
        
        # Exit: reverse signal or weekly trend change
        exit_long = (position == 1 and 
                    (weekly_downtrend_aligned[i] > 0.5 or  # Weekly trend turned down
                     short_entry))  # Opposite signal
        
        exit_short = (position == -1 and 
                     (weekly_uptrend_aligned[i] > 0.5 or  # Weekly trend turned up
                      long_entry))  # Opposite signal
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            signals[i] = position * position_size
    
    return signals

name = "6h_1w_pullback_reversion"
timeframe = "6h"
leverage = 1.0