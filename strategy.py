#!/usr/bin/env python3
"""
6h_4h200_EMA_1d_Trend_Filter_v1
Hypothesis: On 6h timeframe, use 4h EMA200 for primary trend direction and 1d EMA200 for higher timeframe trend confirmation. 
Enter long when price > 4h EMA200 and price > 1d EMA200, short when price < 4h EMA200 and price < 1d EMA200.
Requires volume > 1.3x 20-period average for confirmation to avoid chop. 
Exit when price crosses back below/above the 4h EMA200. 
This dual-timeframe EMA filter should reduce whipsaws in ranging markets while capturing trends in both bull and bear regimes.
Target: 20-40 trades/year by requiring alignment of two timeframes and volume filter.
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
    
    # Get 4h data for EMA200
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA(200)
    ema_len = 200
    ema_4h = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= ema_len:
        # Calculate EMA using Wilder's smoothing (alpha = 1/N)
        alpha = 1.0 / ema_len
        ema_4h[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            ema_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_4h[i-1]
    
    # Align 4h EMA200 to 6h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(200)
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_len:
        alpha = 1.0 / ema_len
        ema_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA200 to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_len, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above both 4h EMA200 and 1d EMA200 + volume confirmation
            if close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i] and volume[i] > 1.3 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below both 4h EMA200 and 1d EMA200 + volume confirmation
            elif close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i] and volume[i] > 1.3 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 4h EMA200 (trend change on lower timeframe)
            if close[i] < ema_4h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 4h EMA200 (trend change on lower timeframe)
            if close[i] > ema_4h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_4h200_EMA_1d_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0