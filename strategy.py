#!/usr/bin/env python3
# 4h_momentum_breakout_v1
# Hypothesis: Combines 4h momentum (ROC) with 1d trend filter (EMA50) and volume confirmation.
# Goes long when ROC(10) > 5, price > EMA50(1d), and volume > 1.5x average volume.
# Goes short when ROC(10) < -5, price < EMA50(1d), and volume > 1.5x average volume.
# Uses momentum to capture breakouts and trend filter to avoid counter-trend trades.
# Designed for 20-50 trades/year on 4h to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_momentum_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Momentum indicator: ROC(10) - Rate of Change over 10 periods
    roc = np.full(n, np.nan)
    for i in range(10, n):
        if close[i-10] != 0:
            roc[i] = (close[i] - close[i-10]) / close[i-10] * 100
    
    # Volume average (20-period) for confirmation
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20, 10)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(roc[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: momentum turns negative or trend filter fails
            if roc[i] <= 0 or close[i] <= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: momentum turns positive or trend filter fails
            if roc[i] >= 0 or close[i] >= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: positive momentum with trend and volume confirmation
            if (roc[i] > 5 and 
                close[i] > ema50_1d_aligned[i] and 
                vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: negative momentum with trend and volume confirmation
            elif (roc[i] < -5 and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals