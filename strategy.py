#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeConfirm
Hypothesis: Weekly pivot levels provide strong institutional support/resistance. 
Breakout above weekly R1 with 1d uptrend and volume confirmation = long.
Breakout below weekly S1 with 1d downtrend and volume confirmation = short.
Uses discrete position sizing (0.25) and minimum holding period (2 bars) to reduce fee churn.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
Weekly pivots from 1w provide structure that aligns with larger timeframe institutional interest.
Works in both bull and bear markets by following the 1d trend direction for breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need warmup for volume median
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (more reliable than standard)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    typical_price = (h_1w + l_1w + c_1w) / 3.0
    hl_range = h_1w - l_1w
    
    # Weekly Camarilla levels
    r1_1w = c_1w + (hl_range * 1.1 / 12.0)
    s1_1w = c_1w - (hl_range * 1.1 / 12.0)
    r4_1w = c_1w + (hl_range * 1.1 / 2.0)  # Strong breakout level
    s4_1w = c_1w - (hl_range * 1.1 / 2.0)  # Strong breakdown level
    
    # Align weekly levels to 6h timeframe (use previous weekly bar's levels)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        r4_val = r4_1w_aligned[i]
        s4_val = s4_1w_aligned[i]
        
        # Long logic: price breaks above weekly R1 with volume spike and 1d uptrend
        # Require break above R4 for stronger confirmation in choppy markets
        long_condition = (close_val > r4_val) and volume_spike[i] and (close_val > ema_val)
        # Short logic: price breaks below weekly S1 with volume spike and 1d downtrend
        # Require break below S4 for stronger confirmation
        short_condition = (close_val < s4_val) and volume_spike[i] and (close_val < ema_val)
        
        # Exit logic: trend reversal
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        # Minimum holding period: 2 bars (12h on 6h chart) to reduce churn
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0