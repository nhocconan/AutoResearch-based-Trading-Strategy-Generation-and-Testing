#!/usr/bin/env python3
name = "6h_WeeklyPivot_Pullback_TrendFilter"
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
    
    # Load weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    # Use previous week's values to avoid look-ahead
    prev_week_high = df_w['high'].shift(1).values
    prev_week_low = df_w['low'].shift(1).values
    prev_week_close = df_w['close'].shift(1).values
    
    # Pivot point and key levels
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h
    pp_aligned = align_htf_to_ltf(prices, df_w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    
    # Load daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    ema_d = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_d_aligned = align_htf_to_ltf(prices, df_d, ema_d)
    
    # Volume filter: > 1.3x 24-period average (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA and sufficient data
    
    for i in range(start_idx, n):
        if np.isnan(ema_d_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or \
           np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Pullback to S1 in weekly uptrend (price above weekly PP) with volume
            if (close[i] > pp_aligned[i] and  # Above weekly pivot = uptrend
                low[i] <= s1_aligned[i] and  # Pullback to S1
                close[i] > ema_d_aligned[i] and  # Above daily EMA50
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Pullback to R1 in weekly downtrend (price below weekly PP) with volume
            elif (close[i] < pp_aligned[i] and  # Below weekly pivot = downtrend
                  high[i] >= r1_aligned[i] and  # Pullback to R1
                  close[i] < ema_d_aligned[i] and  # Below daily EMA50
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Break below S2 or trend change
            if close[i] < s2_aligned[i] or close[i] < ema_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Break above R2 or trend change
            if close[i] > r2_aligned[i] or close[i] > ema_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot pullback strategy on 6h timeframe.
# In weekly uptrend (price above weekly PP), pullbacks to S1 offer buying opportunities with trend continuation.
# In weekly downtrend (price below weekly PP), pullbacks to R1 offer selling opportunities.
# Uses daily EMA(50) as intermediate trend filter and volume confirmation for institutional participation.
# Weekly pivot provides significant support/resistance that institutions respect.
# Target: 15-25 trades/year to minimize fee drag. Position size 0.25 controls drawdown.
# Works in both bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend) markets.