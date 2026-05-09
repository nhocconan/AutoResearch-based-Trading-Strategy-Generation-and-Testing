#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeS
Hypothesis: Use Camarilla pivot levels from daily data for breakout entries.
Only take long breaks above R1 when 1d EMA34 is rising and short breaks below S1 when 1d EMA34 is falling.
Volume spike confirms breakout legitimacy.
Designed for low trade frequency (12-37/year) on 12h timeframe with trend alignment and volume filter.
Works in both bull and bear markets by following 1d trend direction.
"""

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (using previous day's data)
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    R1 = prev_close + (range_ * 1.1 / 12)
    S1 = prev_close - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (available after 1d bar closes)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend: rising EMA34 = uptrend, falling EMA34 = downtrend
    ema_prev = np.roll(ema_34_1d_aligned, 1)
    ema_prev[0] = ema_34_1d_aligned[0]
    trend_up = ema_34_1d_aligned > ema_prev
    trend_down = ema_34_1d_aligned < ema_prev
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike + session
            if close[i] > R1_aligned[i] and trend_up[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 1d downtrend + volume spike + session
            elif close[i] < S1_aligned[i] and trend_down[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below S1 or trend reversal
            if close[i] < S1_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above R1 or trend reversal
            if close[i] > R1_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals