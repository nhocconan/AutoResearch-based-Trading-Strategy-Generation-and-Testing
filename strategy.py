#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly trend: price above/below EMA200
    weekly_uptrend = close_1w[-1] > ema200_1w[-1] if len(close_1w) > 0 else False
    weekly_downtrend = close_1w[-1] < ema200_1w[-1] if len(close_1w) > 0 else False
    
    # Calculate daily pivot points (standard floor trader's pivots)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align weekly trend to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, 
                                              np.full(len(close_1w), weekly_uptrend, dtype=float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, 
                                                np.full(len(close_1w), weekly_downtrend, dtype=float))
    
    # Align daily pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 6-period RSI for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Get current weekly trend
        weekly_uptrend = weekly_uptrend_aligned[i] > 0.5
        weekly_downtrend = weekly_downtrend_aligned[i] > 0.5
        
        # Fade at R3/S3 in ranging markets, breakout at R4/S4 in trending markets
        # R4/R5 and S4/S5 extensions
        r4_1d = r3_1d + (high_1d - low_1d)
        s4_1d = s3_1d - (high_1d - low_1d)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
        
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Market regime: weekly trend strength
        strong_weekly_uptrend = weekly_uptrend and (close[i] > ema200_1w[-1] if len(close_1w) > 0 else False)
        strong_weekly_downtrend = weekly_downtrend and (close[i] < ema200_1w[-1] if len(close_1w) > 0 else False)
        
        # Fade conditions at R3/S3 (counter-trend in ranging markets)
        fade_r3 = close[i] >= r3_aligned[i] and rsi[i] > 70
        fade_s3 = close[i] <= s3_aligned[i] and rsi[i] < 30
        
        # Breakout conditions at R4/S4 (trend continuation)
        breakout_r4 = close[i] > r4_aligned[i] and rsi[i] > 50
        breakdown_s4 = close[i] < s4_aligned[i] and rsi[i] < 50
        
        # Long conditions
        long_condition = False
        if strong_weekly_uptrend:
            # In uptrend, look for breakouts at R4
            long_condition = breakout_r4
        else:
            # In downtrend or ranging, look for fades at S3
            long_condition = fade_s3 and not strong_weekly_downtrend
        
        # Short conditions
        short_condition = False
        if strong_weekly_downtrend:
            # In downtrend, look for breakdowns at S4
            short_condition = breakdown_s4
        else:
            # In uptrend or ranging, look for fades at R3
            short_condition = fade_r3 and not strong_weekly_uptrend
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite signal or RSI extreme reversal
        elif position == 1 and (close[i] < pivot_aligned[i] or rsi[i] < 30):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > pivot_aligned[i] or rsi[i] > 70):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA200_Trend_DailyPivot_FadeBreakout"
timeframe = "6h"
leverage = 1.0