#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and session filter (08-20 UTC)
# Uses prior day's OHLC for Camarilla levels to avoid look-ahead, 4h EMA34 for trend alignment,
# and session filter to reduce noise. Discrete position sizing (0.20) controls fee drag.
# Target: 60-150 total trades over 4 years (15-37/year) by requiring confluence of
# daily breakout level, 4h trend, and active session. Works in bull markets by capturing
# breakouts with trend, works in bear by only taking trend-aligned breaks.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA34_Trend_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate daily Camarilla levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels: 
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4
    r3_1d = close_1d + camarilla_range
    s3_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 1h timeframe (they update daily)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need at least 1 day of prior data)
    start_idx = 24  # 1 day in hours
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above R3 with bullish 4h trend
            if close[i] > r3_1d_aligned[i] and ema_34_4h_aligned[i] > close_4h[-1] if len(close_4h) > 0 else False:
                # Re-check trend using aligned values for bar i
                close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
                if close_4h_aligned[i] > ema_34_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            # Short: Close breaks below S3 with bearish 4h trend
            elif close[i] < s3_1d_aligned[i]:
                close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
                if close_4h_aligned[i] < ema_34_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below S3 (reversal to mean) OR 4h trend turns bearish
            close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
            if close[i] < s3_1d_aligned[i] or close_4h_aligned[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close rises above R3 (reversal to mean) OR 4h trend turns bullish
            close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
            if close[i] > r3_1d_aligned[i] or close_4h_aligned[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals