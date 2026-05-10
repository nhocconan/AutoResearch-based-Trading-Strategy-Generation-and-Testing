#!/usr/bin/env python3
"""
12H_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Uses Camarilla pivot levels from daily data to identify key S1/R1 support/resistance. 
Long when price breaks above R1 with volume confirmation and daily EMA34 uptrend. 
Short when price breaks below S1 with volume confirmation and daily EMA34 downtrend.
Designed for 12h timeframe to capture multi-day swings with low trade frequency (target: 12-37/year).
Works in both bull and bear markets by following daily trend direction, avoiding false signals in ranging markets.
"""

name = "12H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Daily data for Camarilla pivots and EMA34 trend (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Formula: R1 = close + 1.1*(high - low)/12, S1 = close - 1.1*(high - low)/12
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Daily EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume filter: volume > 1.5x 24-period average on 12h chart
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: daily EMA34 slope
        if i > start_idx:
            ema34_prev = ema34_aligned[i-1]
        else:
            ema34_prev = ema34_aligned[i]
        is_uptrend = ema34_aligned[i] > ema34_prev
        is_downtrend = ema34_aligned[i] < ema34_prev
        
        if position == 0:
            # Long entry: price breaks above R1 + volume + daily uptrend
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + volume + daily downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or daily trend turns down
            if (close[i] < s1_aligned[i] or not is_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or daily trend turns up
            if (close[i] > r1_aligned[i] or not is_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals