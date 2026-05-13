#!/usr/bin/env python3
"""
6h_1w_TurningPoint_Bounce
Hypothesis: On 6h timeframe, price bouncing from weekly pivot support/resistance levels 
during strong 1-week trends provides high-probability mean-reversion entries in both 
bull and bear markets. Uses weekly pivot levels as dynamic support/resistance and 
1-week EMA for trend filter. Target: 15-25 trades/year per symbol.
"""

name = "6h_1w_TurningPoint_Bounce"
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
    
    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    
    # Shift weekly data by 1 to use only completed weeks
    wk_high = df_1w['high'].shift(1).values
    wk_low = df_1w['low'].shift(1).values
    wk_close = df_1w['close'].shift(1).values
    
    # Calculate pivot levels
    pivot = (wk_high + wk_low + wk_close) / 3
    r1 = 2 * pivot - wk_low
    s1 = 2 * pivot - wk_high
    r2 = pivot + (wk_high - wk_low)
    s2 = pivot - (wk_high - wk_low)
    r3 = wk_high + 2 * (pivot - wk_low)
    s3 = wk_low - 2 * (wk_high - pivot)
    
    # Align weekly pivot levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # 1-week trend: 21 EMA on weekly close
    wk_close_series = df_1w['close']
    ema21_1w = wk_close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    # Shift EMA to avoid look-ahead (use only completed weeks)
    ema21_1w_shifted = np.roll(ema21_1w, 1)
    ema21_1w_shifted[0] = 0  # First value invalid
    
    # Align weekly EMA to 6h
    ema21_1w_6h = align_htf_to_ltf(prices, df_1w, ema21_1w_shifted)
    
    # Determine weekly trend direction
    uptrend_1w = wk_close > ema21_1w_shifted
    downtrend_1w = wk_close < ema21_1w_shifted
    
    # Align trend to 6h
    uptrend_1w_6h = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_6h = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get current price and aligned weekly levels
        price = close[i]
        pivot_val = pivot_6h[i]
        r1_val = r1_6h[i]
        s1_val = s1_6h[i]
        r2_val = r2_6h[i]
        s2_val = s2_6h[i]
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        uptrend = uptrend_1w_6h[i]
        downtrend = downtrend_1w_6h[i]
        
        if position == 0:
            # LONG: Price near weekly support in uptrend
            # Enter when price touches or slightly breaks S1/S2 but holds above S3
            if uptrend and (price <= s1_val * 1.002) and (price >= s3_val * 0.998):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near weekly resistance in downtrend
            elif downtrend and (price >= r1_val * 0.998) and (price <= r3_val * 1.002):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly pivot or resistance, or trend turns down
            if price >= pivot_val * 0.998 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly pivot or support, or trend turns up
            if price <= pivot_val * 1.002 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals