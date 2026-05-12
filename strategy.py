#!/usr/bin/env python3
# 1d_1w_Chaikin_Money_Flow_Trend
# Hypothesis: Uses weekly Chaikin Money Flow (CMF) as trend filter and daily price action for entries.
# Enters long when price closes above previous day's high with bullish weekly CMF (>0.1).
# Enters short when price closes below previous day's low with bearish weekly CMF (<-0.1).
# Designed for low trade frequency (<50 total trades over 4 years) to minimize fee drift.
# Works in bull markets via CMF>0 and in bear markets via CMF<0, avoiding chop via CMF magnitude filter.

name = "1d_1w_Chaikin_Money_Flow_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Previous day's high/low for breakout logic
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Weekly data for Chaikin Money Flow
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range_1w = high_1w - low_1w
    mf_multiplier = np.where(hl_range_1w != 0, 
                             ((close_1w - low_1w) - (high_1w - close_1w)) / hl_range_1w, 
                             0)
    
    # Money Flow Volume = Money Flow Multiplier * Volume
    mf_volume = mf_multiplier * volume_1w
    
    # Chaikin Money Flow = 20-period sum of MFV / 20-period sum of volume
    mfv_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume_1w).rolling(window=20, min_periods=20).sum().values
    cmf_20 = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    # Align CMF to daily timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1w, cmf_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(cmf_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close above prev day high + bullish CMF (>0.1)
            if (close[i] > prev_high[i] and 
                cmf_aligned[i] > 0.1):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below prev day low + bearish CMF (<-0.1)
            elif (close[i] < prev_low[i] and 
                  cmf_aligned[i] < -0.1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below prev day low OR CMF turns bearish (<0)
            if (close[i] < prev_low[i]) or \
               (cmf_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above prev day high OR CMF turns bullish (>0)
            if (close[i] > prev_high[i]) or \
               (cmf_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals