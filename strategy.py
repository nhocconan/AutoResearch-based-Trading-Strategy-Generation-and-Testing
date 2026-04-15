#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Pivot Breakout with Volume Confirmation and Volume Weighted Average Price (VWAP) Filter
# Uses weekly pivot points (S1, S2, R1, R2) as support/resistance levels.
# Long when price breaks above R1 with volume > 1.5x average and price > VWAP.
# Short when price breaks below S1 with volume > 1.5x average and price < VWAP.
# Weekly pivots calculated from prior week's high/low/close to avoid look-ahead.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (avoid look-ahead)
    # Shift by 1 to use only completed weeks
    prev_week_high = np.roll(df_weekly['high'].values, 1)
    prev_week_low = np.roll(df_weekly['low'].values, 1)
    prev_week_close = np.roll(df_weekly['close'].values, 1)
    
    # First values have no prior week
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    # Pivot point calculation
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Calculate daily VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size: 25% of capital
    
    for i in range(50, n):  # Start after sufficient data for VWAP
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vwap[i])):
            continue
        
        # Long entry: price breaks above R1 with volume confirmation and price > VWAP
        if (close[i] > r1_aligned[i] and
            volume[i] > 1.5 * np.median(window := volume[max(0, i-20):i+1]) and
            close[i] > vwap[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below S1 with volume confirmation and price < VWAP
        elif (close[i] < s1_aligned[i] and
              volume[i] > 1.5 * np.median(window := volume[max(0, i-20):i+1]) and
              close[i] < vwap[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout to opposite S2/R2 level
        elif position == 1 and close[i] < s2_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r2_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Pivot_Breakout_Volume_VWAP"
timeframe = "1d"
leverage = 1.0