#!/usr/bin/env python3
# 1d_KAMA_Trend_With_Weekly_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction on daily.
# Weekly trend filter (price > weekly EMA20) ensures trades align with higher timeframe momentum.
# Volume surge (current volume > 1.5x 20 EMA) confirms institutional interest.
# Works in bull markets via trend-following entries and in bear via short signals when price < KAMA.
# Low trade frequency expected due to triple confirmation (KAMA trend, weekly filter, volume).
# Target: 10-25 trades/year on 1d timeframe.

name = "1d_KAMA_Trend_With_Weekly_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_len=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = abs(pd.Series(close).diff(er_len))
    volatility = pd.Series(close).diff().abs().rolling(window=er_len, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = pd.Series(close).copy()
    for i in range(1, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close[i] - kama.iloc[i-1])
    return kama.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Calculate daily KAMA for trend
    daily_close = prices['close'].values
    kama_val = kama(daily_close, er_len=10, fast=2, slow=30)
    
    # Volume filter: current volume > 1.5x 20-period EMA
    volume = prices['volume'].values
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (20) + KAMA (30) + vol EMA (20)
    start_idx = max(20, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_ema20_aligned[i]) or 
            np.isnan(kama_val[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA AND price > weekly EMA20 AND volume surge
            if daily_close[i] > kama_val[i] and daily_close[i] > weekly_ema20_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND price < weekly EMA20 AND volume surge
            elif daily_close[i] < kama_val[i] and daily_close[i] < weekly_ema20_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < KAMA OR price < weekly EMA20
            if daily_close[i] < kama_val[i] or daily_close[i] < weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA OR price > weekly EMA20
            if daily_close[i] > kama_val[i] or daily_close[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals