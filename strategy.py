#!/usr/bin/env python3
"""
6h_6hr_ChaikinMoneyFlow_RangeBreakout_1dTrend
Hypothesis: CMF confirms institutional flow direction; breakout from 20-period high/low with CMF confirmation and 1-day trend filter captures sustainable moves. Works in bull (breakouts with +CMF) and bear (breakdowns with -CMF). Target: 15-30 trades/year (60-120 total).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Chaikin Money Flow (20)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)
    mfv = mfm * volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / vol_sum
    cmf = np.where(vol_sum == 0, 0, cmf)
    
    # Donchian channel (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(cmf[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(ema_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        cmf_val = cmf[i]
        hh = highest_high[i]
        ll = lowest_low[i]
        ema_trend = ema_1d_6h[i]
        
        if position == 0:
            # Long: break above 20-period high with positive CMF and uptrend
            if price > hh and cmf_val > 0.05 and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low with negative CMF and downtrend
            elif price < ll and cmf_val < -0.05 and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retrace to midpoint OR CMF turns negative
            midpoint = (hh + ll) / 2
            if price < midpoint or cmf_val < -0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retrace to midpoint OR CMF turns positive
            midpoint = (hh + ll) / 2
            if price > midpoint or cmf_val > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_6hr_ChaikinMoneyFlow_RangeBreakout_1dTrend"
timeframe = "6h"
leverage = 1.0