# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_ElderRay_TrendFollowing_v1
Primary timeframe: 6h, HTF: 1d
Elder Ray (Bull/Bear Power) from 1d combined with 6h trend filter.
Long when 1d Bull Power > 0 and 6h close > 6h EMA50.
Short when 1d Bear Power < 0 and 6h close < 6h EMA50.
Exit when Elder Power reverses or price crosses EMA50.
Designed for low trade frequency and works in both bull and bear markets by using 1d Elder Ray for regime and 6h EMA for entry timing.
"""

import numpy as np
import pandas as pd
from mtrand import seed
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_TrendFollowing_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA50 for trend filter and entry timing
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Elder Ray calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h timeframe (wait for 1d bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema50[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d Bull Power positive AND 6h close above EMA50
            if bull_power_aligned[i] > 0 and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d Bear Power negative AND 6h close below EMA50
            elif bear_power_aligned[i] < 0 and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 1d Bear Power turns negative OR price crosses below EMA50
            if bear_power_aligned[i] < 0 or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 1d Bull Power turns positive OR price crosses above EMA50
            if bull_power_aligned[i] > 0 or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals