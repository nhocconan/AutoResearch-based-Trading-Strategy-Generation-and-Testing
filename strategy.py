# %%
#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align weekly trend to daily
    weekly_trend_raw = (close_1w > ema200_1w).astype(float)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_raw)
    
    # Align daily pivot levels to daily (self-alignment for same timeframe)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate daily RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Get current weekly trend
        weekly_uptrend = weekly_trend_aligned[i] > 0.5
        
        # Fade at R1/S1 with RSI extremes
        fade_r1 = close[i] >= r1_aligned[i] and rsi[i] > 70
        fade_s1 = close[i] <= s1_aligned[i] and rsi[i] < 30
        
        # Long conditions
        long_condition = False
        if weekly_uptrend:
            # In uptrend, look for pullbacks to S1
            long_condition = fade_s1
        else:
            # In downtrend, look for bounces at R1 (counter-trend bounce)
            long_condition = fade_r1
        
        # Short conditions
        short_condition = False
        if weekly_uptrend:
            # In uptrend, look for rejections at R1
            short_condition = fade_r1
        else:
            # In downtrend, look for breakdowns from S1
            short_condition = fade_s1
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite signal or RSI mean reversion
        elif position == 1 and (close[i] < pivot_aligned[i] or rsi[i] > 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > pivot_aligned[i] or rsi[i] < 30):
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

name = "1d_WeeklyEMA200_Trend_DailyPivot_Fade"
timeframe = "1d"
leverage = 1.0