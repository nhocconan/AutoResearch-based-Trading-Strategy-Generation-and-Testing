#!/usr/bin/env python3
# 4H_PriceAction_Breakout_Volume_Trend
# Hypothesis: Price action breakout of previous 1d high/low with volume confirmation and trend filter on 4h.
# Enters long when price breaks above previous 1d high with volume > 2x average and 4h close > 4h EMA50.
# Enters short when price breaks below previous 1d low with volume > 2x average and 4h close < 4h EMA50.
# Exits when price returns to the opposite level (previous 1d low for long, previous 1d high for short).
# Uses 4h EMA50 for trend to filter false breakouts and works in both bull/bear markets.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25 to minimize fee drag.

name = "4H_PriceAction_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for previous day high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d high and low
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align to 4h timeframe (available after 1d bar closes)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or np.isnan(ema_50_4h[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        price_above_ema = close[i] > ema_50_4h[i]
        price_below_ema = close[i] < ema_50_4h[i]
        
        if position == 0:
            # Long entry: break above previous 1d high with volume and uptrend
            if (close[i] > prev_high_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: break below previous 1d low with volume and downtrend
            elif (close[i] < prev_low_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to previous 1d low
            if close[i] < prev_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to previous 1d high
            if close[i] > prev_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals