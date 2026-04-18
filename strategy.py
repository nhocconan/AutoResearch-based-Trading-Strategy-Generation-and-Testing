#!/usr/bin/env python3
"""
1d Weekly Pivot R1/S1 Breakout with Volume Spike and Weekly EMA Trend
Hypothesis: Weekly pivot levels (R1, S1) from weekly chart act as strong support/resistance.
Breakouts beyond these levels with volume confirmation and weekly EMA trend filter capture momentum.
Designed for 7-25 trades/year on 1d timeframe with low trade frequency to minimize fee drag.
Works in bull/bear markets by requiring volume spike and weekly EMA trend filter.
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using standard formula
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Using previous week's data to avoid look-ahead
    weekly_high = df_w['high']
    weekly_low = df_w['low']
    weekly_close = df_w['close']
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Shift by 1 to use previous week's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_w, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1_prev)
    
    # Get weekly data for EMA trend filter
    ema_34_w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_w, ema_34_w)
    
    # Volume spike: 2x 20-period average on 1d
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
            # Long: break above R1 with volume spike and price above weekly EMA (uptrend)
            if price > r1_val and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and price below weekly EMA (downtrend)
            elif price < s1_val and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below weekly EMA
            if price <= s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above weekly EMA
            if price >= r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_EMA"
timeframe = "1d"
leverage = 1.0