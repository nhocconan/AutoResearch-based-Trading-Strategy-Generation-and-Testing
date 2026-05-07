#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: Breakouts above/below 20-period Donchian channel on 4h, filtered by 1d EMA trend and volume confirmation.
Long when price breaks above upper band in 1d uptrend with volume > 1.5x average; short when breaks below lower band in 1d downtrend with volume > 1.5x average.
Exit when price returns to midline (median of Donchian channel).
Designed for 4h to capture trend continuations with moderate frequency (target 30-60 trades/year).
"""

name = "4h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) on 4h
    period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    midline = (upper + lower) / 2
    
    # Get 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d close aligned for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, period)  # Warmup for 1d EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend determination
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: break above upper band in 1d uptrend with volume confirmation
            if (close[i] > upper[i] and 
                trend_1d_up and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band in 1d downtrend with volume confirmation
            elif (close[i] < lower[i] and 
                  trend_1d_down and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midline
            if close[i] <= midline[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midline
            if close[i] >= midline[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals