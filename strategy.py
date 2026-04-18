#!/usr/bin/env python3
"""
12h Daily Pivot R1/S1 Breakout with Volume Spike and Daily EMA Trend
Hypothesis: Daily pivot levels (R1, S1) from daily chart act as strong support/resistance.
Breakouts beyond these levels with volume confirmation and daily EMA trend filter capture momentum.
Designed for 12-37 trades/year on 12h timeframe with low trade frequency to minimize fee drag.
Works in bull/bear markets by requiring volume spike and daily EMA trend filter.
"""

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
    
    # Get daily data for pivot calculation (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points using standard formula
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Using previous day's data to avoid look-ahead
    daily_high = df_d['high']
    daily_low = df_d['low']
    daily_close = df_d['close']
    
    pivot = (daily_high + daily_low + daily_close) / 3
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Shift by 1 to use previous day's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1_prev)
    
    # Get daily data for EMA trend filter (same df_d)
    # Daily EMA34 for trend filter
    ema_34_d = pd.Series(df_d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_d, ema_34_d)
    
    # Volume spike: 2x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and price above daily EMA (uptrend)
            if price > r1_val and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and price below daily EMA (downtrend)
            elif price < s1_val and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below daily EMA
            if price <= s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above daily EMA
            if price >= r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_DailyPivot_R1S1_Breakout_Volume_EMA"
timeframe = "12h"
leverage = 1.0