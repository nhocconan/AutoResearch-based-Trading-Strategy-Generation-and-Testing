#!/usr/bin/env python3

"""
Hypothesis: 6-hour Weekly Pivot Pullback with Daily Trend Filter and Volume Confirmation.
Trades pullbacks to weekly pivot levels in the direction of the daily EMA trend during volume expansion.
Uses weekly pivot as institutional reference point and daily EMA for trend alignment to work in both bull and bear markets.
Designed for low trade frequency (15-30 trades/year) by requiring confluence of weekly level, daily trend, and volume spike.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot levels (P, R1, S1, R2, S2) for given arrays."""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = pivot + range_val
    s1 = pivot - range_val
    r2 = pivot + 2 * range_val
    s2 = pivot - 2 * range_val
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot levels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot, weekly_r1, weekly_s1, weekly_r2, weekly_s2 = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Load daily data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_30[i]
        
        if position == 0 and vol_spike:
            # Long: pullback to weekly S1/S2 support with uptrend bias
            if (close[i] > weekly_s1_aligned[i] and close[i] < weekly_pivot_aligned[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: pullback to weekly R1/R2 resistance with downtrend bias
            elif (close[i] < weekly_r1_aligned[i] and close[i] > weekly_pivot_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses weekly pivot or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below weekly S1 or closes below daily EMA
                if close[i] < weekly_s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above weekly R1 or closes above daily EMA
                if close[i] > weekly_r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_Pullback_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0