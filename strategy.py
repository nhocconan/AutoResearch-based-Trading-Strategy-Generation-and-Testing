#!/usr/bin/env python3
"""
6h Weekly Pivot Camarilla Fade + Daily EMA34 Trend + Volume Spike
Hypothesis: Weekly Camarilla pivot levels (H3/L3) act as key support/resistance.
Price fading from these levels with daily EMA34 trend alignment and volume spike
captures institutional reversals. Works in bull markets (fades at resistance) and
bear markets (bounces at support). 6h timeframe targets 12-37 trades/year to
minimize fee drag while capturing meaningful swings.
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
    
    # Weekly data for Camarilla pivot calculation (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Camarilla pivot levels
    # HLC from previous weekly bar (already completed due to get_htf_data)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla calculations
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    range_val = weekly_high - weekly_low
    
    # H3 and L3 levels (primary fade levels)
    h3 = pivot + (range_val * 1.1 / 4.0)
    l3 = pivot - (range_val * 1.1 / 4.0)
    
    # Align weekly levels to 6h timeframe (no extra delay needed for pivot points)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Daily EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly data, daily EMA, and volume MA
    start_idx = max(34, 20) + 10  # extra for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Fade conditions: price near H3 (sell) or near L3 (buy)
        # Use 0.5% buffer around levels to avoid whipsaws
        near_h3 = abs(curr_close - h3_aligned[i]) / h3_aligned[i] < 0.005
        near_l3 = abs(curr_close - l3_aligned[i]) / l3_aligned[i] < 0.005
        
        if position == 0:
            # Look for entry signals - require: fade level + volume spike + daily EMA34 trend alignment
            long_entry = near_l3 and vol_spike and (curr_close > ema_34_1d_aligned[i])
            short_entry = near_h3 and vol_spike and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price moves back to midpoint or trend changes
            midpoint = (h3_aligned[i] + l3_aligned[i]) / 2.0
            if curr_close > midpoint or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price moves back to midpoint or trend changes
            midpoint = (h3_aligned[i] + l3_aligned[i]) / 2.0
            if curr_close < midpoint or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyCamarilla_H3L3_Fade_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0