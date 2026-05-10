#!/usr/bin/env python3
# 4h_3_Bar_Reversal_1dTrend_Volume_Spike
# Hypothesis: 3-bar reversal pattern (3 consecutive closes in same direction) with 1d EMA trend filter and volume spike.
# Works in bull/bear markets by aligning with daily trend. Targets 20-40 trades/year to minimize fee drag.
# Uses price action only - no lagging indicators - for robust signals.

name = "4h_3_Bar_Reversal_1dTrend_Volume_Spike"
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
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 2  # Warmup for daily EMA, volume MA, plus 2 for lookback
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # 3-bar reversal pattern: 3 consecutive closes moving in same direction
        # Bullish: 3 higher closes
        bullish_pattern = (close[i] > close[i-1] and 
                          close[i-1] > close[i-2] and 
                          close[i-2] > close[i-3])
        # Bearish: 3 lower closes
        bearish_pattern = (close[i] < close[i-1] and 
                          close[i-1] < close[i-2] and 
                          close[i-2] < close[i-3])
        
        if position == 0:
            # Long entry: bullish 3-bar pattern + uptrend + volume spike
            if bullish_pattern and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish 3-bar pattern + downtrend + volume spike
            elif bearish_pattern and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below the low of the 3-bar pattern or trend reversal
            pattern_low = min(close[i-3], close[i-2], close[i-1])
            if close[i] < pattern_low or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above the high of the 3-bar pattern or trend reversal
            pattern_high = max(close[i-3], close[i-2], close[i-1])
            if close[i] > pattern_high or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals