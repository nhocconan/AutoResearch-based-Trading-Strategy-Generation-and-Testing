#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla Pivot + Daily Volume Spike + Weekly Trend Filter
# Hypothesis: Camarilla levels act as institutional support/resistance. Long at L3 with daily volume spike and weekly uptrend, short at H3 with volume spike and weekly downtrend.
# Uses 12h timeframe for lower frequency, 1d for volume confirmation, 1w for trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
name = "12h_camarilla_pivot_1d_volume_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Pivot = (H + L + C) / 3
    # H4 = Pivot + 1.5 * (H - L)
    # L4 = Pivot - 1.5 * (H - L)
    # H3 = Pivot + 1.125 * (H - L)
    # L3 = Pivot - 1.125 * (H - L)
    # H2 = Pivot + 0.75 * (H - L)
    # L2 = Pivot - 0.75 * (H - L)
    # H1 = Pivot + 0.5 * (H - L)
    # L1 = Pivot - 0.5 * (H - L)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    rang = high_1d - low_1d
    h3 = pivot + 1.125 * rang
    l3 = pivot - 1.125 * rang
    h4 = pivot + 1.5 * rang
    l4 = pivot - 1.5 * rang
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to ensure previous day data is available
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current 12h volume > 1.5x daily average volume
        vol_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Weekly trend filter: price above/below weekly EMA21
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches L4 (stop) or H3 (take profit) or trend changes
            if close[i] <= l4_aligned[i] or close[i] >= h3_aligned[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches H4 (stop) or L3 (take profit) or trend changes
            if close[i] >= h4_aligned[i] or close[i] <= l3_aligned[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price at L3 with volume spike and weekly uptrend
            if abs(close[i] - l3_aligned[i]) < 0.001 * close[i] and vol_spike and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: price at H3 with volume spike and weekly downtrend
            elif abs(close[i] - h3_aligned[i]) < 0.001 * close[i] and vol_spike and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals