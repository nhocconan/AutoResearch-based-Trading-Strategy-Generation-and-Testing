# 1d_1w_Camarilla_Touch_V1
# Hypothesis: Daily Camarilla pivot touch with weekly trend filter. Long when price touches daily S3/S4 in weekly uptrend, short when touches R3/R4 in weekly downtrend.
# Weekly trend uses EMA(8) vs EMA(21) to avoid whipsaw. Target low trade frequency: ~60 total over 4 years (~15/year) to minimize fee drag.
# Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Touch_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA trend: fast EMA(8) > slow EMA(21) = uptrend
    close_1w = df_1w['close'].values
    ema_fast = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_slow = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = ema_fast > ema_slow
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate Camarilla levels (standard formula)
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Camarilla levels: H5, H4, H3, L3, L4, L5
    camarilla_h4 = prev_close + range_val * 1.1 / 2
    camarilla_l4 = prev_close - range_val * 1.1 / 2
    camarilla_h3 = prev_close + range_val * 1.1 / 4
    camarilla_l3 = prev_close - range_val * 1.1 / 4
    
    # Align Camarilla levels to daily (already daily, but use for consistency)
    camarilla_h4_array = np.full(len(df_1d), camarilla_h4)
    camarilla_l4_array = np.full(len(df_1d), camarilla_l4)
    camarilla_h3_array = np.full(len(df_1d), camarilla_h3)
    camarilla_l3_array = np.full(len(df_1d), camarilla_l3)
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_array)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_array)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_array)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_array)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter
        is_uptrend = weekly_uptrend_aligned[i]
        
        # Entry conditions: touch Camarilla levels in direction of trend
        # Long: touch L3/L4 in uptrend
        # Short: touch H3/H4 in downtrend
        long_touch = (low[i] <= camarilla_l3_aligned[i] or low[i] <= camarilla_l4_aligned[i]) and is_uptrend
        short_touch = (high[i] >= camarilla_h3_aligned[i] or high[i] >= camarilla_h4_aligned[i]) and not is_uptrend
        
        # Exit conditions: opposite touch or reversal to midline
        # Midline approximated as (H4 + L4)/2
        midline = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
        long_exit = high[i] >= midline and position == 1
        short_exit = low[i] <= midline and position == -1
        
        # Signal logic
        if long_touch and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_touch and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals