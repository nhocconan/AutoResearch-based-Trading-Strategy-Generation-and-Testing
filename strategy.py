#!/usr/bin/env python3
name = "1d_WeeklyPivot_KAMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly pivot points from weekly data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot levels
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike detection: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 20)  # Wait for KAMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and KAMA uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            kama_uptrend = kama[i] > kama[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and kama_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and KAMA downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not kama_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below pivot or KAMA turns down
            if close[i] < pp_aligned[i] or kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above pivot or KAMA turns up
            if close[i] > pp_aligned[i] or kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily weekly pivot breakout with KAMA trend and volume confirmation
# - Weekly pivot points (S1/R1) act as dynamic support/resistance levels
# - Breakout above S1 with volume in KAMA uptrend = long opportunity
# - Breakdown below R1 with volume in KAMA downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation
# - KAMA adapts to market conditions: fast in trends, slow in ranges
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to weekly pivot (PP) or trend changes
# - Position size 0.25 targets 15-25 trades/year, avoiding fee drag
# - Weekly pivot provides structure that works across market regimes