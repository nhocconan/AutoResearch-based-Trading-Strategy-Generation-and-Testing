#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Breakouts at 1d Camarilla R3 (long) and S3 (short) levels on 12h timeframe, filtered by 1d EMA34 trend and volume spike, with position size 0.25. Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong momentum moves in both bull and bear markets.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    s3 = close_1d - (range_1d * 1.25000)  # S3 level
    r3 = close_1d + (range_1d * 1.25000)  # R3 level
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # 1d trend filter: EMA 34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema_34_1d_aligned[i]
        is_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R3 + volume spike + 1d uptrend
            if (close[i] > r3_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S3 + volume spike + 1d downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S3 (opposite side)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R3 (opposite side)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals