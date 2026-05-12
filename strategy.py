#!/usr/bin/env python3
"""
6H_ElderRay_PowerIndex_1D_Trend_Filter
Hypothesis: Elder Ray Power Index (Bull/Bear Power) with 1-day trend filter.
Long when Bull Power > 0 and price above 1-day EMA50; Short when Bear Power < 0 and price below 1-day EMA50.
Uses Elder Ray's raw power (Close - EMA13 for Bull, EMA13 - Close for Bear) to capture institutional buying/selling pressure.
Filters trades to only those aligned with the daily trend, reducing whipsaws in chop.
Designed for ~15-30 trades/year on 6h to minimize fee drift while capturing sustained moves.
Works in bull/bear markets by requiring alignment with higher-timeframe trend.
"""
name = "6H_ElderRay_PowerIndex_1D_Trend_Filter"
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
    
    # Elder Ray: Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema13
    bear_power = ema13 - close
    
    # 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA13 warmup
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power positive AND price above 1D EMA50 (uptrend)
            if bull_power[i] > 0 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power positive AND price below 1D EMA50 (downtrend)
            elif bear_power[i] > 0 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative OR price breaks below 1D EMA50
            if bull_power[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns negative OR price breaks above 1D EMA50
            if bear_power[i] <= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals