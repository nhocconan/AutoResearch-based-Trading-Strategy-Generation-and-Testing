#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_S1R1_Breakout_Trend"
timeframe = "1h"
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
    
    # 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Daily bias filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: 4h uptrend, daily uptrend, price above S1
            if in_session and ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and close[i] > s1_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, daily downtrend, price below R1
            elif in_session and ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and close[i] < r1_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: 4h trend breaks or price returns to S1
            if ema_21_4h_aligned[i] <= ema_21_4h_aligned[i-1] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: 4h trend breaks or price returns to R1
            if ema_21_4h_aligned[i] >= ema_21_4h_aligned[i-1] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla S1/R1 breakout with 4h and daily trend alignment
# - Uses 1h for entry timing, 4h for intermediate trend, 1d for bias and structure
# - Daily Camarilla S1/R1 from previous day act as key support/resistance
# - Requires alignment of 4h EMA(21) and daily EMA(34) trends for entry
# - Session filter (8-20 UTC) reduces noise during low-liquidity hours
# - Works in bull markets (buy S1 breaks in uptrends) and bear markets (sell R1 breaks in downtrends)
# - Position size 0.20 limits risk; targets ~20-40 trades/year to avoid fee drag
# - Exits when trend breaks or price returns to S1/R1 levels
# - Designed for BTC/ETH; avoids overtrading through multi-timeframe confluence