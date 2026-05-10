#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_1dTrend_Volume
# Hypothesis: Weekly pivot levels from 1w timeframe provide strong support/resistance.
# Breakouts above R1 or below S1 with 1d EMA trend alignment and volume confirmation
# capture institutional flow. Works in bull/bear by following weekly structure.
# Target: 15-25 trades/year (~60-100 total over 4 years) to stay within optimal trade frequency for 6h.

name = "6h_Weekly_Pivot_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Weekly pivot levels (R1, S1) - calculated from prior week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use previous week's OHLC for pivot calculation (avoid look-ahead)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot_point - prev_week_low
    s1 = 2 * pivot_point - prev_week_high
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # 1d EMA trend filter (34-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # already datetime64[ms], .hour works
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1, 1d EMA uptrend, volume confirmation, session active
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, 1d EMA downtrend, volume confirmation, session active
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below weekly S1 OR 1d EMA turns down
            if (close[i] < s1_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above weekly R1 OR 1d EMA turns up
            if (close[i] > r1_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals