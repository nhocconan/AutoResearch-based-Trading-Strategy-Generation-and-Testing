#!/usr/bin/env python3
# 4h_1d_vwap_trend_v1
# Hypothesis: Trade with VWAP trend on 4h using 1d trend filter. Long when price > VWAP and price > 1d EMA(50),
# short when price < VWAP and price < 1d EMA(50). Exit when price crosses back across VWAP.
# Uses volume-weighted average price as dynamic support/resistance with trend filter to avoid counter-trend trades.
# Target: 20-40 trades/year (80-160 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_vwap_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    cum_pv = np.zeros(n)
    cum_vol = np.zeros(n)
    
    cum_pv[0] = pv[0]
    cum_vol[0] = volume[0]
    
    for i in range(1, n):
        cum_pv[i] = cum_pv[i-1] + pv[i]
        cum_vol[i] = cum_vol[i-1] + volume[i]
    
    vwap = np.zeros(n)
    vwap[0] = typical_price[0]  # First period VWAP is typical price
    
    for i in range(1, n):
        if cum_vol[i] > 0:
            vwap[i] = cum_pv[i] / cum_vol[i]
        else:
            vwap[i] = vwap[i-1]
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d, dtype=float)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(vwap[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP
            if close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP
            if close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above VWAP and above 1d EMA(50)
            if close[i] > vwap[i] and close[i] > ema_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price below VWAP and below 1d EMA(50)
            elif close[i] < vwap[i] and close[i] < ema_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals