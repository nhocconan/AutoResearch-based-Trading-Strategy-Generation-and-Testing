#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyPivot_CamarillaBreakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    pp_w = (weekly_high + weekly_low + weekly_close) / 3.0
    r1_w = (2 * pp_w) - weekly_low
    s1_w = (2 * pp_w) - weekly_high
    
    # Calculate daily Camarilla levels (H-L range based)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_range = daily_high - daily_low
    camarilla_h4 = daily_close + 1.5 * daily_range
    camarilla_l4 = daily_close - 1.5 * daily_range
    
    # Align weekly pivots and daily Camarilla to 12h timeframe
    pp_w_12h = align_htf_to_ltf(prices, df_1w, pp_w)
    r1_w_12h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_12h = align_htf_to_ltf(prices, df_1w, s1_w)
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: spike above 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(pp_w_12h[i]) or np.isnan(r1_w_12h[i]) or np.isnan(s1_w_12h[i]) or
            np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC)
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above weekly S1 and daily Camarilla L4, 1d uptrend, volume breakout
            if (close[i] > s1_w_12h[i] and 
                close[i] > camarilla_l4_12h[i] and 
                close[i] > ema_50_12h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly R1 and daily Camarilla H4, 1d downtrend, volume breakdown
            elif (close[i] < r1_w_12h[i] and 
                  close[i] < camarilla_h4_12h[i] and 
                  close[i] < ema_50_12h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly S2 or daily Camarilla L4 or trend reversal
            if close[i] < s1_w_12h[i] or close[i] < camarilla_l4_12h[i] or close[i] < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly R2 or daily Camarilla H4 or trend reversal
            if close[i] > r1_w_12h[i] or close[i] > camarilla_h4_12h[i] or close[i] > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals