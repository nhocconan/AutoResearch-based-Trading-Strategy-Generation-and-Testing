#!/usr/bin/env python3
"""
6h_weekly_pivot_breakout_v1
Hypothesis: 6h price breaks above/below weekly pivot levels (R4/S4) with volume confirmation and weekly trend filter.
In bull markets: buy breakouts above weekly resistance with uptrend.
In bear markets: sell breakdowns below weekly support with downtrend.
Weekly pivots provide strong institutional levels; volume confirms participation.
Target: 15-30 trades/year to avoid overtrading on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly pivot levels (calculated from prior week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = weekly_high + 3 * (pp - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Weekly trend filter: price vs weekly EMA(13)
    ema_13 = np.full(len(weekly_close), np.nan)
    for i in range(13, len(weekly_close)):
        ema_13[i] = np.mean(weekly_close[i-13:i])  # Simple MA for efficiency
    
    ema_13_aligned = align_htf_to_ltf(prices, df_1w, ema_13)
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_13_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price_above_r4 = close[i] > r4_aligned[i]
        price_below_s4 = close[i] < s4_aligned[i]
        weekly_uptrend = close[i] > ema_13_aligned[i]
        weekly_downtrend = close[i] < ema_13_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below S4 or weekly trend turns down
            if price_below_s4 or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above R4 or weekly trend turns up
            if price_above_r4 or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: breakout above R4 with volume and weekly uptrend
            if (price_above_r4 and 
                vol_ratio > 1.5 and 
                weekly_uptrend):
                position = 1
                signals[i] = 0.25
            # Short: breakdown below S4 with volume and weekly downtrend
            elif (price_below_s4 and 
                  vol_ratio > 1.5 and 
                  weekly_downtrend):
                position = -1
                signals[i] = -0.25
    
    return signals