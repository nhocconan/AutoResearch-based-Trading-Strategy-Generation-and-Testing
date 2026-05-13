#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation.
# Elder Ray measures bull/bear power as (High - EMA13) and (EMA13 - Low). 
# Uses 12h EMA for trend direction to avoid counter-trend trades. Volume confirms strength.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ElderRay_12hTrend_Volume"
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
    volume = prices['volume'].values
    
    # Load 12H data ONCE for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 for Elder Ray (both 6h and 12h)
    close_6h_series = pd.Series(close)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    close_12h_series = pd.Series(close_12h)
    ema13_12h = close_12h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components for 6h
    bull_power = high - ema13_6h  # High - EMA13
    bear_power = ema13_6h - low   # EMA13 - Low
    
    # Align 12h EMA to 6h timeframe for trend filter
    ema13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    
    # Volume filter: current volume > 20-period average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema13_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA13
        price_above_ema = close[i] > ema13_12h_aligned[i]
        price_below_ema = close[i] < ema13_12h_aligned[i]
        
        if position == 0:
            # LONG: Strong bull power with uptrend and volume
            if (bull_power[i] > 0) and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Strong bear power with downtrend and volume
            elif (bear_power[i] > 0) and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear power becomes strong or trend changes
            if (bear_power[i] > 0) or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull power becomes strong or trend changes
            if (bull_power[i] > 0) or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals