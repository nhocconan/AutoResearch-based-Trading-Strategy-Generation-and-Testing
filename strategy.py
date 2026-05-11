#!/usr/bin/env python3
name = "6h_Chaikin_Oscillator_Range_Bound_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Chaikin Money Flow (CMF) - 21 period
    mf_multiplier = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    mf_volume = mf_multiplier * volume
    cmf = pd.Series(mf_volume).rolling(window=21, min_periods=21).sum() / pd.Series(volume).rolling(window=21, min_periods=21).sum()
    cmf = cmf.values
    
    # Chaikin Oscillator = EMA(3, CMF) - EMA(10, CMF)
    cmf_series = pd.Series(cmf)
    ema3 = cmf_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = cmf_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3 - ema10
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Align Chaikin Oscillator to 6t
    chaikin_aligned = align_htf_to_ltf(prices, df_1d, chaikin_osc)
    
    signals = np.zeros(n)
    
    start_idx = max(21, 10, 34)
    
    for i in range(start_idx, n):
        if np.isnan(chaikin_aligned[i]) or np.isnan(ema34_aligned[i]):
            if i > 0:
                signals[i] = signals[i-1]
            else:
                signals[i] = 0.0
            continue
        
        # Mean reversion in ranging markets: buy when Chaikin oversold, sell when overbought
        # Only in uptrend (price above EMA34) for longs, downtrend for shorts
        if close[i] > ema34_aligned[i]:  # Uptrend
            if chaikin_aligned[i] < -0.15 and chaikin_aligned[i-1] >= -0.15:
                signals[i] = 0.25  # Long
            elif chaikin_aligned[i] > 0.15 and chaikin_aligned[i-1] <= 0.15:
                signals[i] = 0.0   # Exit long
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        elif close[i] < ema34_aligned[i]:  # Downtrend
            if chaikin_aligned[i] > 0.15 and chaikin_aligned[i-1] <= 0.15:
                signals[i] = -0.25  # Short
            elif chaikin_aligned[i] < -0.15 and chaikin_aligned[i-1] >= -0.15:
                signals[i] = 0.0    # Exit short
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        else:
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals