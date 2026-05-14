#!/usr/bin/env python3
"""
12h_VolumeBreakout_1dTrend_HighVolume
Hypothesis: Volume breakouts above 1.8x 20-period average with 1d EMA34 trend filter capture momentum moves.
Works in bull/bear by filtering with 1d EMA34 trend. Targets 15-25 trades/year on 12h to minimize fee drag.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: volume breakout above close with uptrend
            if close[i] > close[i-1] and vol_confirm_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: volume breakdown below close with downtrend
            elif close[i] < close[i-1] and vol_confirm_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: volume breakout down or trend turns down
            if close[i] < close[i-1] and vol_confirm_val or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: volume breakdown up or trend turns up
            if close[i] > close[i-1] and vol_confirm_val or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_VolumeBreakout_1dTrend_HighVolume"
timeframe = "12h"
leverage = 1.0