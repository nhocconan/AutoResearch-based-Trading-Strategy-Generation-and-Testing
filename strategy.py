#!/usr/bin/env python3
"""
Hypothesis: 1-hour ORB (Opening Range Breakout) with 4h trend filter and 1d volume confirmation.
Long when price breaks above first 1-hour opening range (00:00-01:00 UTC) with 4h EMA50 rising and 1d volume > 1.5x 20-day average.
Short when price breaks below opening range low with 4h EMA50 falling and 1d volume > 1.5x 20-day average.
Exit when price returns to the opening range midpoint.
Designed for low trade frequency (1-2 trades per day max) by requiring multiple confirmations and using the daily opening range as structure.
Works in both bull and bear markets by following the 4h trend.
"""

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
    open_time = prices['open_time'].values
    
    # Pre-calculate hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data for volume confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day volume 20-period average for confirmation
    vol_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    
    # Calculate daily opening range (00:00-01:00 UTC) for each day
    # We'll track the high and low of the first hour of each day
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    or_mid = np.full(n, np.nan)
    
    # Track opening range for each day
    day_start_idx = None
    day_high = None
    day_low = None
    
    for i in range(n):
        hour = hours[i]
        # 00:00 UTC marks start of day
        if hour == 0:
            day_start_idx = i
            day_high = high[i]
            day_low = low[i]
        elif day_start_idx is not None:
            # Update high/low during first hour (00:00-01:00 UTC)
            if hour < 1:
                day_high = max(day_high, high[i])
                day_low = min(day_low, low[i])
            # At 01:00 UTC, first hour complete - set opening range for the day
            elif hour == 1:
                or_high[i] = day_high
                or_low[i] = day_low
                or_mid[i] = (day_high + day_low) / 2.0
                # Reset for next day
                day_start_idx = None
                day_high = None
                day_low = None
    
    # Forward fill opening range values for the entire day
    for i in range(1, n):
        if np.isnan(or_high[i]):
            or_high[i] = or_high[i-1]
            or_low[i] = or_low[i-1]
            or_mid[i] = or_mid[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for EMA50
        # Skip if data not ready
        if (np.isnan(or_high[i]) or np.isnan(or_low[i]) or np.isnan(or_mid[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_20d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation: 1d volume > 1.5x 20-day average
        vol_confirmed = df_1d['volume'].values[min(i//24, len(df_1d)-1)] > 1.5 * vol_20d_aligned[i] if i >= 24 else False
        
        if position == 0 and in_session:
            # Long: Price breaks above OR high with 4h EMA50 rising and volume confirmed
            if (close[i] > or_high[i] and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and vol_confirmed):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below OR low with 4h EMA50 falling and volume confirmed
            elif (close[i] < or_low[i] and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and vol_confirmed):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Price returns to opening range midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below OR midpoint
                if close[i] < or_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above OR midpoint
                if close[i] > or_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_ORB_4hEMA50_Trend_1dVolume"
timeframe = "1h"
leverage = 1.0