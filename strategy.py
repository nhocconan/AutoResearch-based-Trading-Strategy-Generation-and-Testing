#!/usr/bin/env python3
"""
6h_Chaikin_Money_Flow_Breakout_1dTrend
Hypothesis: Use Chaikin Money Flow (CMF) to detect institutional accumulation/distribution on 6h timeframe.
Enter long when CMF crosses above +0.15 with price above 1d EMA50, short when CMF crosses below -0.15 with price below 1d EMA50.
Exit when CMF returns to neutral zone (-0.1 to +0.1). This captures smart money moves while avoiding whipsaws.
Works in both bull and bear markets by following institutional flow rather than price alone.
Timeframe: 6h
"""

name = "6h_Chaikin_Money_Flow_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for EMA50 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Chaikin Money Flow (CMF) on 6h data
    # CMF = ADV(20) / Volume(20) where ADV = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # Prevent division by zero
    
    money_flow_multiplier = ((close - low) - (high - close)) / hl_range
    money_flow_volume = money_flow_multiplier * volume
    
    # 20-period sums for CMF
    mfv_sum = pd.Series(money_flow_volume).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vol_sum = np.where(vol_sum == 0, 1e-10, vol_sum)  # Prevent division by zero
    
    cmf = mfv_sum / vol_sum

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after CMF warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cmf[i]) or 
            np.isnan(money_flow_multiplier[i]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CMF crosses above +0.15 AND price above 1d EMA50
            if (cmf[i] > 0.15 and cmf[i-1] <= 0.15 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CMF crosses below -0.15 AND price below 1d EMA50
            elif (cmf[i] < -0.15 and cmf[i-1] >= -0.15 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF returns to neutral zone (above -0.1)
            if cmf[i] > -0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF returns to neutral zone (below +0.1)
            if cmf[i] < 0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals