#!/usr/bin/env python3
# 1H_Camarilla_Breakout_4hTrend_VolumeConfirmation
# Hypothesis: 4h timeframe provides better trend reliability than 1h alone. 
# Using 4h EMA50 for trend direction and 1h for entry timing reduces whipsaw.
# Breakouts occur at 1h Camarilla R1/S1 levels with volume > 1.5x average.
# Volume confirmation filters false breakouts. Session filter (08-20 UTC) reduces noise.
# Target: 15-35 trades/year per symbol.

name = "1H_Camarilla_Breakout_4hTrend_VolumeConfirmation"
timeframe = "1h"
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
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA 50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h Camarilla pivot levels from previous 1h bar
    # Calculate from previous bar's high/low/close
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first value to NaN as there's no previous bar
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    range_1h = high_prev - low_prev
    # Avoid division by zero or invalid ranges
    range_1h = np.where(range_1h <= 0, np.nan, range_1h)
    
    # Camarilla formulas
    s1 = close_prev - (range_1h * 1.08333)
    s3 = close_prev - (range_1h * 1.25000)
    r1 = close_prev + (range_1h * 1.08333)
    r3 = close_prev + (range_1h * 1.25000)
    
    # Volume filter: volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need enough history for volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any required values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(s1[i]) or np.isnan(s3[i]) or 
            np.isnan(r1[i]) or np.isnan(r3[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        is_uptrend = close[i] > ema_50_4h_aligned[i]
        is_downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R1 or R3 + volume confirmation + 4h uptrend
            if ((close[i] > r1[i] or close[i] > r3[i]) and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.20
                position = 1
            # Short entry: Price breaks below S1 or S3 + volume confirmation + 4h downtrend
            elif ((close[i] < s1[i] or close[i] < s3[i]) and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S1 (opposite side)
            if close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price crosses above R1 (opposite side)
            if close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals