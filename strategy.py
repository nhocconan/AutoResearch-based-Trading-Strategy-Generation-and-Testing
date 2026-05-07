#!/usr/bin/env python3
# 6h_ElderRay_BullPower_BearPower_1dTrend_Filter
# Hypothesis: Uses Elder Ray indicator (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h timeframe.
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) AND 1d EMA34 trend up.
# Short when Bull Power < 0 and Bear Power > 0 (bearish momentum) AND 1d EMA34 trend down.
# Designed for 6h to capture momentum shifts with trend filter, reducing whipsaws in ranging markets.
# Works in both bull and bear markets via 1d trend filter aligning with higher timeframe bias.

name = "6h_ElderRay_BullPower_BearPower_1dTrend_Filter"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d trend up
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND 1d trend down
            elif bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema_34_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Loss of bullish momentum (Bear Power >= 0) or trend change
            if bear_power[i] >= 0 or close[i] < ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Loss of bearish momentum (Bull Power <= 0) or trend change
            if bull_power[i] <= 0 or close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals