#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1wTrend_Filter
# Hypothesis: Elder Ray (Bull/Bear Power) on 6h with 1-week trend filter.
# Bull Power = EMA(13) - Low; Bear Power = High - EMA(13).
# Long when Bull Power > 0 and weekly trend up; Short when Bear Power > 0 and weekly trend down.
# Weekly trend defined by price above/below 20-period EMA on weekly chart.
# Works in bull (trend following) and bear (counter-trend via Elder Ray extremes) via trend filter.
# Target: 15-30 trades/year to avoid fee drag.

name = "6h_ElderRay_BullBearPower_1wTrend_Filter"
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
    
    # === 1-week Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 20-period EMA on 1w for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Elder Ray (Bull/Bear Power) on 6h ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = ema_13 - low  # EMA(13) - Low
    bear_power = high - ema_13  # High - EMA(13)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend direction
        weekly_up = close[i] > ema_20_1w_aligned[i]
        weekly_down = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # LONG: Bull Power positive and weekly trend up
            if (bull_power[i] > 0 and weekly_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power positive and weekly trend down
            elif (bear_power[i] > 0 and weekly_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or weekly trend changes
            if (bull_power[i] <= 0 or not weekly_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns negative or weekly trend changes
            if (bear_power[i] <= 0 or not weekly_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals