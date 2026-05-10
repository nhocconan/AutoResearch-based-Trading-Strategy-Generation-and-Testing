#!/usr/bin/env python3
"""
4H_KAMA_Trend_Filter_Volume_Breakout
Hypothesis: KAMA(14) trend direction combined with volume spikes and Bollinger Band squeezes
captures explosive moves in both bull and bear markets. The KAMA adapts to market noise,
reducing whipsaws, while volume confirmation ensures momentum. Bollinger Band squeeze
identifies low volatility periods preceding breakouts. Designed for low trade frequency
(<30/year) to minimize fee decay while maintaining edge across regimes.
"""

name = "4H_KAMA_Trend_Filter_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d data for Bollinger Band squeeze detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Bollinger Bands on 1d close
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width < 20-period mean width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    
    # KAMA on 4h close (adaptive to market noise)
    # Efficiency Ratio: |close - close[10]| / sum(|close[i] - close[i-1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will fix below
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: volume > 2.0x 20-period average (tight)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for KAMA
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(vol_threshold[i]) or np.isnan(bb_squeeze_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: price relative to KAMA
        is_uptrend = close[i] > kama[i]
        is_downtrend = close[i] < kama[i]
        
        if position == 0:
            # Long entry: KAMA uptrend + volume spike + Bollinger Band squeeze (breakout from low vol)
            if (is_uptrend and 
                volume[i] > vol_threshold[i] and 
                bb_squeeze_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA downtrend + volume spike + Bollinger Band squeeze
            elif (is_downtrend and 
                  volume[i] > vol_threshold[i] and 
                  bb_squeeze_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals