#!/usr/bin/env python3
"""
12h_ema_bounce_volume_v1
Hypothesis: Price bounces off EMA(50) on 12h with volume confirmation. Works in both bull and bear markets
by capturing mean-reversion moves within the trend. Uses daily trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_bounce_volume_v1"
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
    
    # Daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align daily EMA to 12h timeframe
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h EMA50 for bounce signals
    ema50_12h_calc = pd.Series(close).ewm(span=50, adjust=False).mean().values
    
    # 20-period volume average on 12h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema50_12h[i]) or np.isnan(ema50_12h_calc[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA
            if close[i] < ema50_12h_calc[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above EMA
            if close[i] > ema50_12h_calc[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long when price touches EMA from below in uptrend
            if (close[i] <= ema50_12h_calc[i] * 1.005 and  # Allow small tolerance
                close[i] > ema50_12h_calc[i] * 0.995 and
                vol_confirm and
                ema50_12h[i] > ema50_12h[max(0, i-1)]):  # Daily EMA rising
                position = 1
                signals[i] = 0.25
            # Short when price touches EMA from above in downtrend
            elif (close[i] >= ema50_12h_calc[i] * 0.995 and
                  close[i] < ema50_12h_calc[i] * 1.005 and
                  vol_confirm and
                  ema50_12h[i] < ema50_12h[max(0, i-1)]):  # Daily EMA falling
                position = -1
                signals[i] = -0.25
    
    return signals