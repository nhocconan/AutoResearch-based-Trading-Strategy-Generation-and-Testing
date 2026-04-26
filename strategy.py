#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter
Hypothesis: 4-hour Camarilla R3/S3 breakout with 1d EMA34 trend filter and choppiness regime filter.
Enters long when price breaks above R3 with bullish 1d trend and low chop (trending market).
Enters short when price breaks below S3 with bearish 1d trend and low chop.
Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Designed for 75-200 total trades over 4 years.
Works in both bull and bear markets by following the 1d trend direction only, avoiding range-bound whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla pivot levels (R3, S3) on 1d timeframe
    # Use prior completed 1d bar to avoid look-ahead
    df_1d = get_htf_data(prices, '1d')
    
    # Prior 1d bar's OHLC for Camarilla calculation (shifted by 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_open = np.roll(df_1d['open'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    prior_open[0] = np.nan
    
    # Calculate pivot point and Camarilla levels
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_hl = prior_high - prior_low
    r3 = pivot + range_hl * 1.1 / 4
    s3 = pivot - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Load 1d data for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate choppiness index on 1d timeframe to avoid ranging markets
    atr_period = 14
    # True Range calculation
    tr1 = np.abs(np.roll(high, 1) - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    # Chop = 100 * log10(sum(atr) / (max(high) - min(low))) / log10(period)
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    sum_atr = pd.Series(atr_1d).rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(atr_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need 1d shift + 34-period EMA + 14-period chop)
    start_idx = 1 + max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish 1d trend + chop < 61.8 (trending market)
        if close[i] > r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and chop_aligned[i] < 61.8:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish 1d trend + chop < 61.8 (trending market)
        elif close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and chop_aligned[i] < 61.8:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s3_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r3_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter"
timeframe = "4h"
leverage = 1.0