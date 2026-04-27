#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_1dTrend_Volume
Hypothesis: Weekly pivots provide strong support/resistance. Price breaking above weekly R1 with 1d EMA34 uptrend and volume spike captures bullish moves; breaking below weekly S1 with 1d EMA34 downtrend and volume spike captures bearish moves. Weekly context filters out noise, works in bull via R1 breakouts and bear via S1 breakdowns. Targets ~20 trades/year on 6h to minimize fee drag.
"""

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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using previous week)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Weekly Pivot Point: PP = (H + L + C) / 3
    # Weekly R1 = (2 * PP) - L
    # Weekly S1 = (2 * PP) - H
    pp_w = (high_w + low_w + close_w) / 3
    r1_w = (2 * pp_w) - low_w
    s1_w = (2 * pp_w) - high_w
    
    # Align weekly R1/S1 to 6h timeframe (use previous week's levels)
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = r1_w_aligned[i]
        s1 = s1_w_aligned[i]
        ema_trend = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with uptrend and volume spike
            if close[i] > r1 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly S1 with downtrend and volume spike
            elif close[i] < s1 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below weekly S1 or trend turns down
            if close[i] < s1 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above weekly R1 or trend turns up
            if close[i] > r1 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Direction_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0