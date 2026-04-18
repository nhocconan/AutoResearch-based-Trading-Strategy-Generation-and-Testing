#!/usr/bin/env python3
"""
6h_LongTermTrend_With_1D_Pullback
Hypothesis: Use 1-day EMA(50) as long-term trend filter (proven to avoid 2022 whipsaws), combined with 6-hour pullback to EMA(20) and volume confirmation. In bull markets, go long when price pulls back to EMA(20) in uptrend; in bear markets, go short when price rallies to EMA(20) in downtrend. EMA(50) provides smooth trend direction, reducing false signals during chop. Targets 15-25 trades/year by requiring EMA(50) alignment, EMA(20) touch, and volume > 1.5x average. Works in both bull and bear by only trading with the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily close
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * 0.0377 + ema_50_1d[i-1] * (1 - 0.0377)
    
    # Align EMA(50) to 6h timeframe (wait for bar close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(20) on 6h close for entry timing
    ema_20 = np.full(n, np.nan)
    if n >= 20:
        ema_20[19] = np.mean(close[:20])
        for i in range(20, n):
            ema_20[i] = close[i] * 0.0909 + ema_20[i-1] * (1 - 0.0909)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # need both EMAs
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price pulls back to EMA(20) in uptrend (close > EMA50_1d)
            if (close[i] >= ema_20[i] * 0.998 and close[i] <= ema_20[i] * 1.002 and
                close[i] > ema_50_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price rallies to EMA(20) in downtrend (close < EMA50_1d)
            elif (close[i] >= ema_20[i] * 0.998 and close[i] <= ema_20[i] * 1.002 and
                  close[i] < ema_50_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below EMA(20) or trend changes
            if close[i] < ema_20[i] * 0.995 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above EMA(20) or trend changes
            if close[i] > ema_20[i] * 1.005 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_LongTermTrend_With_1D_Pullback"
timeframe = "6h"
leverage = 1.0