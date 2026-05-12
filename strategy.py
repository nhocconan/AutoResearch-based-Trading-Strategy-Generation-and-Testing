#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_1wTrend
# Hypothesis: On 1d timeframe, use weekly Camarilla pivot levels (R1/S1) for breakout entries with 1w EMA50 trend filter.
# Enter long when price closes above R1 with 1w EMA50 uptrend.
# Enter short when price closes below S1 with 1w EMA50 downtrend.
# Exit when price crosses the 1w EMA50 (trend reversal).
# Targets 15-25 trades/year to minimize fee drag while capturing multi-week trends in both bull and bear markets.
# Fixed position sizing: 0.30 for all entries to avoid unnecessary signal changes.

name = "1d_Camarilla_R1S1_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data for Camarilla pivot calculation and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and range
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Camarilla R1 and S1 levels
    r1 = weekly_pivot + weekly_range * 1.083
    s1 = weekly_pivot - weekly_range * 1.083
    
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema1w_trend = ema50_1w_aligned[i]
        
        if position == 0:
            # LONG: Price closes above R1 with 1w uptrend
            if close[i] > r1_val and close[i] > ema1w_trend:
                signals[i] = 0.30
                position = 1
            # SHORT: Price closes below S1 with 1w downtrend
            elif close[i] < s1_val and close[i] < ema1w_trend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1w EMA50 (trend reversal)
            if close[i] < ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above 1w EMA50 (trend reversal)
            if close[i] > ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals