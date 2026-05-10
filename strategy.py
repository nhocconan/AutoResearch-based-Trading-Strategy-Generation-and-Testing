#!/usr/bin/env python3
# 6h_Elder_Ray_1dTrend_Filter
# Hypothesis: Elder Ray index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1-day trend filter.
# Goes long when Bull Power > 0 and 1-day trend is up (close > EMA34), short when Bear Power > 0 and 1-day trend is down.
# Uses 13-period EMA for Elder Ray calculation on 6h timeframe. Filters trades with 1-day EMA34 trend.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Targets 12-30 trades per year on 6h timeframe with position size 0.25.

name = "6h_Elder_Ray_1dTrend_Filter"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA(13) for Elder Ray on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13)  # Warmup for 1d EMA and 6h EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: Bull Power positive AND 1-day uptrend
            if bull_power[i] > 0 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power positive AND 1-day downtrend
            elif bear_power[i] > 0 and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR 1-day trend turns down
            if bull_power[i] <= 0 or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative OR 1-day trend turns up
            if bear_power[i] <= 0 or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals