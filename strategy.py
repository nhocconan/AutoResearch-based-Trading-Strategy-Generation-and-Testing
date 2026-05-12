#!/usr/bin/env python3
# 12h_Chaikin_Oscillator_1dTrend_Volume
# Hypothesis: On 12h timeframe, Chaikin Oscillator crossing zero with volume confirmation
# and 1d EMA trend filter captures momentum shifts in both bull and bear markets.
# Designed for low trade frequency (~15-30/year) to minimize fee drag.

name = "12h_Chaikin_Oscillator_1dTrend_Volume"
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
    
    # === 1d Trend Filter (Higher Timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on daily for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Chaikin Oscillator (3,10) on 12h ===
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # ADL = cumulative sum of Money Flow Volume
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    
    # Avoid division by zero in range
    ranges = high - low
    mfm = np.where(ranges > 0, ((close - low) - (high - close)) / ranges, 0.0)
    mfv = mfm * volume
    
    # ADL (Accumulation/Distribution Line)
    adl = np.cumsum(mfv)
    
    # Chaikin Oscillator: EMA(3, ADL) - EMA(10, ADL)
    adl_series = pd.Series(adl)
    ema_3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema_10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema_3_adl - ema_10_adl
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure Chaikin, EMA, and ADL are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chaikin[i]) or 
            np.isnan(ema_3_adl[i]) or np.isnan(ema_10_adl[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # LONG: Chaikin crosses above zero (buying pressure) + uptrend bias
            if chaikin[i] > 0 and chaikin[i-1] <= 0 and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin crosses below zero (selling pressure) + downtrend bias
            elif chaikin[i] < 0 and chaikin[i-1] >= 0 and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Chaikin crosses below zero or trend changes
            if chaikin[i] < 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin crosses above zero or trend changes
            if chaikin[i] > 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals