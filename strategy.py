#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1w trend filter (price > weekly EMA34 for long, < for short) and volume confirmation (1.8x 20-period MA).
# Enters long when price breaks above R3 with bullish weekly trend and volume spike.
# Enters short when price breaks below S3 with bearish weekly trend and volume spike.
# Exits when price reverts to the daily pivot point (PP) from 1d timeframe.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: Camarilla breakout + weekly trend + volume spike.
# Camarilla levels provide institutional support/resistance; weekly EMA34 filter ensures alignment with major trend; volume spike confirms institutional participation.

name = "6h_Camarilla_R3S3_Breakout_1wTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    pp = (high + low + close) / 3
    r1 = pp + (range_val * 1.1 / 12)
    r2 = pp + (range_val * 1.1 / 6)
    r3 = pp + (range_val * 1.1 / 4)
    r4 = pp + (range_val * 1.1 / 2)
    s1 = pp - (range_val * 1.1 / 12)
    s2 = pp - (range_val * 1.1 / 6)
    s3 = pp - (range_val * 1.1 / 4)
    s4 = pp - (range_val * 1.1 / 2)
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w close
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get 1d data for exit (daily pivot point)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Calculate daily pivot point: PP = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(100, 34)  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(pp_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 6h bar
        pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high[i], low[i], close[i])
        
        if position == 0:
            # LONG: Price breaks above R3 with bullish weekly trend and volume spike
            if close[i] > r3 and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with bearish weekly trend and volume spike
            elif close[i] < s3 and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to daily pivot point (mean reversion to PP)
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to daily pivot point (mean reversion to PP)
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals