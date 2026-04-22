#!/usr/bin/env python3

"""
Hypothesis: 6h ADX trend strength combined with 1d Williams %R mean reversion.
In strong trends (ADX > 25), pullbacks to oversold/overbought levels (Williams %R < -80 or > -20)
offer high-probability continuation entries. Works in both bull and bear markets by following
the trend direction from ADX. Low trade frequency achieved by requiring both trend strength
and extreme momentum readings.
"""

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
    
    # 6h ADX for trend strength (14-period)
    plus_dm = np.diff(high, prepend=high[0])
    minus_dm = np.diff(low, prepend=low[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr = np.maximum(
        np.maximum(high - low, np.abs(high - np.roll(low, 1))),
        np.abs(low - np.roll(high, 1))
    )
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d Williams %R for mean reversion (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if np.isnan(adx[i]) or np.isnan(williams_r_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: strong trend + Williams %R oversold (< -80)
            if strong_trend and williams_r_aligned[i] < -80:
                signals[i] = 0.25
                position = 1
            # Short: strong trend + Williams %R overbought (> -20)
            elif strong_trend and williams_r_aligned[i] > -20:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend weakness or Williams %R returns to neutral range
            exit_signal = False
            
            if position == 1:
                # Exit long: trend weakens or Williams %R rises above -50
                if adx[i] <= 25 or williams_r_aligned[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: trend weakens or Williams %R falls below -50
                if adx[i] <= 25 or williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ADX_TrendStrength_1d_WilliamsR_MeanReversion"
timeframe = "6h"
leverage = 1.0