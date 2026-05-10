#!/usr/bin/env python3
"""
4H_HTF_Trend_Touch_Signal
Hypothesis: Touch of 4h price to 1d/200 EMA or 1w/50 EMA with volume confirmation and HTF trend alignment captures directional moves in both bull and bear markets. Uses minimal conditions to keep trade frequency low (<25/year) and reduce fee drag. Long when price touches rising EMA from below; short when price touches falling EMA from above.
"""

name = "4H_HTF_Trend_Touch_Signal"
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
    open_price = prices['open'].values
    
    # 1d data for EMA 200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1w data for EMA 50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA 200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1w EMA 50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine EMA slopes using prior values (avoid look-ahead)
        ema_200_prev = ema_200_1d_aligned[i-1] if i > 0 else ema_200_1d_aligned[i]
        ema_50_prev = ema_50_1w_aligned[i-1] if i > 0 else ema_50_1w_aligned[i]
        
        ema_200_rising = ema_200_1d_aligned[i] > ema_200_prev
        ema_200_falling = ema_200_1d_aligned[i] < ema_200_prev
        ema_50_rising = ema_50_1w_aligned[i] > ema_50_prev
        ema_50_falling = ema_50_1w_aligned[i] < ema_50_prev
        
        if position == 0:
            # Long: price touches rising 1d EMA200 from below OR rising 1w EMA50 from below
            touch_1d = (low[i] <= ema_200_1d_aligned[i] * 1.001 and  # within 0.1%
                       high[i-1] < ema_200_1d_aligned[i-1] if i > 0 else True)  # was below
            touch_1w = (low[i] <= ema_50_1w_aligned[i] * 1.001 and
                       high[i-1] < ema_50_1w_aligned[i-1] if i > 0 else True)
            
            if ((touch_1d and ema_200_rising) or (touch_1w and ema_50_rising)) and \
               volume[i] > vol_threshold[i] and \
               close[i] > open_price[i]:  # bullish candle confirmation
                signals[i] = 0.25
                position = 1
            # Short: price touches falling 1d EMA200 from above OR falling 1w EMA50 from above
            elif ((low[i] >= ema_200_1d_aligned[i] * 0.999 and  # within 0.1%
                   low[i-1] > ema_200_1d_aligned[i-1] if i > 0 else True) or  # was above
                  (low[i] >= ema_50_1w_aligned[i] * 0.999 and
                   low[i-1] > ema_50_1w_aligned[i-1] if i > 0 else True)) and \
                  volume[i] > vol_threshold[i] and \
                  close[i] < open_price[i]:  # bearish candle confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA
            if close[i] < ema_200_1d_aligned[i] * 0.995 or close[i] < ema_50_1w_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA
            if close[i] > ema_200_1d_aligned[i] * 1.005 or close[i] > ema_50_1w_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals