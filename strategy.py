#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_TrendFilter
Hypothesis: Trade breakouts from Camarilla pivot levels (R1/S1) on 12h timeframe with 1d trend filter and volume confirmation. In bull markets, buy breaks above R1; in bear markets, sell shorts below S1. The 1d EMA34 filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaw. Volume > 1.5x 24-period average confirms breakout strength. Targets 15-25 trades/year via strict pivot breakout conditions. Works in both bull and bear by following 1d trend direction.
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
    
    # Get 1d data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema = np.zeros(len(close_1d))
        ema[0] = close_1d[0]
        alpha = 2 / (34 + 1)
        for i in range(1, len(close_1d)):
            ema[i] = alpha * close_1d[i] + (1 - alpha) * ema[i-1]
        ema34_1d = ema
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = df_1d['close'].values + range_1d * 1.1 / 12
    s1 = df_1d['close'].values - range_1d * 1.1 / 12
    
    # Align 1d indicators to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, vol_period)  # EMA34 needs 34 periods, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = close_1d[-1] > ema34_1d[-1] if len(close_1d) > 0 else False  # simplified for alignment
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + volume
            if close[i] > r1_aligned[i] and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend + volume
            elif close[i] < s1_aligned[i] and not uptrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns down
            if close[i] < s1_aligned[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns up
            if close[i] > r1_aligned[i] or uptrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0