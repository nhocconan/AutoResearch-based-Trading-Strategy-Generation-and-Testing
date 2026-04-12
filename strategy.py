#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_cci_trend_v1
# Uses weekly CCI trend filter (CCI > 100 = bullish, CCI < -100 = bearish) combined with
# daily CCI pullback entries on 6h chart. Long when weekly trend bullish and daily CCI
# pulls back from overbought (< -100 then crosses back above -100). Short when weekly
# trend bearish and daily CCI pulls back from oversold (> 100 then crosses back below 100).
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in trending markets via trend filter and pullback entries.

name = "6h_1d_1w_cci_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily CCI (20-period)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    tp_ma = typical_price.rolling(window=20, min_periods=20).mean()
    tp_md = typical_price.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci_daily = (typical_price - tp_ma) / (0.015 * tp_md)
    cci_daily = cci_daily.values
    
    # Calculate weekly CCI for trend filter (20-period)
    tp_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    tp_ma_1w = tp_1w.rolling(window=20, min_periods=20).mean()
    tp_md_1w = tp_1w.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci_weekly = (tp_1w - tp_ma_1w) / (0.015 * tp_md_1w)
    cci_weekly = cci_weekly.values
    
    # Align daily and weekly data to 6h timeframe
    cci_daily_aligned = align_htf_to_ltf(prices, df_1d, cci_daily)
    cci_weekly_aligned = align_htf_to_ltf(prices, df_1w, cci_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(cci_daily_aligned[i]) or np.isnan(cci_weekly_aligned[i]):
            signals[i] = 0.0
            continue
        
        cci_d = cci_daily_aligned[i]
        cci_w = cci_weekly_aligned[i]
        
        # Long signal: weekly trend bullish (CCI > 100) and daily CCI crosses above -100 from below
        if cci_w > 100 and cci_d > -100 and i > 50 and cci_daily_aligned[i-1] <= -100:
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25  # hold
        # Short signal: weekly trend bearish (CCI < -100) and daily CCI crosses below 100 from above
        elif cci_w < -100 and cci_d < 100 and i > 50 and cci_daily_aligned[i-1] >= 100:
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25  # hold
        # Exit conditions: CCI returns to neutral zone (-100 to 100)
        elif position == 1 and cci_d >= -100 and cci_d <= 100:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_d >= -100 and cci_d <= 100:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals