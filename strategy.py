#!/usr/bin/env python3
# 6H_Weekly_Pivot_Reversal_With_Daily_Trend
# Hypothesis: Mean-revert at weekly pivot extremes (R4/S4) when daily trend is strong.
# Short at weekly R4 when daily EMA50 is above price (bearish context).
# Long at weekly S4 when daily EMA50 is below price (bullish context).
# Exit when price crosses back to weekly pivot point (PP).
# Uses weekly pivot levels for structure and daily trend for bias.
# Target: 15-25 trades/year per symbol. Works in bull/bear by fading extremes with trend filter.

name = "6H_Weekly_Pivot_Reversal_With_Daily_Trend"
timeframe = "6h"
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
    
    # Daily indicators
    close_s = pd.Series(close)
    ema50_daily = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's HLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # R4 = PP + 3*(H - L)  (aggressive resistance)
    r4 = pp + 3.0 * (weekly_high - weekly_low)
    # S4 = PP - 3*(H - L)  (aggressive support)
    s4 = pp - 3.0 * (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for daily EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_daily[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter short at weekly R4 when daily EMA50 is above price (bearish bias)
            if close[i] >= r4_aligned[i] and ema50_daily[i] > close[i]:
                signals[i] = -0.25
                position = -1
            # Enter long at weekly S4 when daily EMA50 is below price (bullish bias)
            elif close[i] <= s4_aligned[i] and ema50_daily[i] < close[i]:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long when price crosses back to weekly PP
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses back to weekly PP
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals