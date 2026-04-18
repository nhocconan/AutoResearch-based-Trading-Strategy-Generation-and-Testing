#!/usr/bin/env python3
"""
6h_LongOnly_Momentum_Volume
Hypothesis: In the 6h timeframe, price momentum combined with volume spikes captures
continuation moves in both bull and bear markets. Long-only with volatility-based exits
reduces whipsaw and limits trade frequency to avoid fee drag. Uses 1-week trend filter
to avoid counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly 50-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.zeros_like(close_1w)
    ema_50_1w[:] = np.nan
    if len(close_1w) >= 50:
        ema = np.zeros(len(close_1w))
        alpha = 2 / (50 + 1)
        ema[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema[i] = alpha * close_1w[i] + (1 - alpha) * ema[i-1]
        ema_50_1w = ema
    
    # Align weekly EMA to 6h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h momentum: price change over 3 periods (~18h)
    mom = np.zeros_like(close)
    mom[:] = np.nan
    for i in range(3, n):
        mom[i] = (close[i] - close[i-3]) / close[i-3]
    
    # 6h volume spike: current volume > 2x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_ma[:] = np.nan
    for i in range(n):
        if i < 20:
            if i >= 0:
                vol_ma[i] = np.mean(volume[0:i+1])
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(mom[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish weekly trend, positive momentum, volume spike
            if (close[i] > ema_50_1w_aligned[i] and  # price above weekly EMA
                mom[i] > 0.005 and                 # minimum 0.5% momentum
                vol_spike[i]):                     # volume confirmation
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit: momentum turns negative or volume dies
            if mom[i] < -0.002 or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "6h_LongOnly_Momentum_Volume"
timeframe = "6h"
leverage = 1.0