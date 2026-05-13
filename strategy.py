#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_Trend_Filter
Hypothesis: Weekly pivot levels (R4/S4) act as strong support/resistance. Breakouts above R4 or below S4 with weekly trend alignment and volume confirmation capture major moves. Works in bull markets via breakouts and bear markets via breakdowns. Target: 15-35 trades/year per symbol.
"""

name = "6h_WeeklyPivot_Breakout_Trend_Filter"
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
    
    # 6-period EMA for trend
    ema_6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    uptrend = close > ema_6
    downtrend = close < ema_6
    
    # Weekly high/low/close for pivot calculation (using actual weekly data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: R4 = R3 + (R2 - R1), S4 = S3 - (S2 - S1)
    # Standard formula: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H, R2 = P+(H-L), S2 = P-(H-L)
    # R3 = H+2(P-L), S3 = L-2(H-P), R4 = R3+(H-L), S4 = S3-(H-L)
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    pivot = (wk_high + wk_low + wk_close) / 3
    r1 = 2 * pivot - wk_low
    s1 = 2 * pivot - wk_high
    r2 = pivot + (wk_high - wk_low)
    s2 = pivot - (wk_high - wk_low)
    r3 = wk_high + 2 * (pivot - wk_low)
    s3 = wk_low - 2 * (wk_high - pivot)
    r4 = r3 + (wk_high - wk_low)
    s4 = s3 - (wk_high - wk_low)
    
    # Align weekly pivot levels to 6t
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Weekly trend: price above/below weekly pivot
    weekly_uptrend = wk_close > pivot
    weekly_downtrend = wk_close < pivot
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        w_uptrend = weekly_uptrend_aligned[i]
        w_downtrend = weekly_downtrend_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R4, weekly uptrend, volume confirmation
            if close[i] > r4 and w_uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S4, weekly downtrend, volume confirmation
            elif close[i] < s4 and w_downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below weekly pivot or trend turns down
            if close[i] < pivot or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above weekly pivot or trend turns up
            if close[i] > pivot or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals