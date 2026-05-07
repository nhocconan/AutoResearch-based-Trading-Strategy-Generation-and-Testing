#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) on 1h combined with 4h EMA trend filter and volume confirmation captures high-probability breakouts. Camarilla levels provide institutional support/resistance; 4h EMA ensures alignment with higher timeframe momentum; volume confirms breakout strength. Session filter (08-20 UTC) reduces noise. Target: 60-150 trades over 4 years.
"""
name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pivot points from previous day (using daily data for Camarilla)
    # For 1h chart, we use previous day's OHLC to calculate today's Camarilla levels
    # We'll calculate daily pivots and align to 1h
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot: (H + L + C) / 3
    # Camarilla levels:
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    # We'll use R1/S1 for breakout
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Previous day's values for today's levels
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    # First day will have NaN from roll, we'll handle
    
    # Calculate Camarilla R1 and S1 for each day
    camarilla_r1 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 12
    camarilla_s1 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 12
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and pivots
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + 4h downtrend + volume
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price returns to the opposite Camarilla level (mean reversion)
            if position == 1:
                if close[i] <= camarilla_s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] >= camarilla_r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals