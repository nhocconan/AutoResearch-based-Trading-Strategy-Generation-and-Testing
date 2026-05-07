#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_Trend_Filter
# Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) 
# combined with 12h trend filter (EMA34) to capture momentum in both bull and bear markets.
# Long when Bull Power > 0 and price above 12h EMA34 (uptrend).
# Short when Bear Power > 0 and price below 12h EMA34 (downtrend).
# Uses 13-period EMA for Elder Ray calculation as standard.
# Designed for low trade frequency (~15-30/year) to minimize fee drag and work in both bull and bear markets.
# Timeframe: 6h, HTF: 12h for trend filter.

timeframe = "6h"
name = "6h_ElderRay_BullBearPower_Trend_Filter"
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
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = EMA13 - Low
    bear_power = ema_13 - low
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 13 to ensure we have EMA13
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive (bullish momentum) and price above 12h EMA34 (uptrend)
            if bull_power[i] > 0 and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive (bearish momentum) and price below 12h EMA34 (downtrend)
            elif bear_power[i] > 0 and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative OR price breaks below 12h EMA34
            if bull_power[i] <= 0 or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns negative OR price breaks above 12h EMA34
            if bear_power[i] <= 0 or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals