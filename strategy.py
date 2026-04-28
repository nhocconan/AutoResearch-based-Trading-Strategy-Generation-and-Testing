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
    
    # Get daily data for weekly pivot and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from daily data (weekly high/low/close)
    # Resample to weekly: get Monday open, week high, week low, Friday close
    # We'll approximate using rolling windows for simplicity
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Weekly support/resistance levels
    r1 = weekly_pivot + (weekly_range * 1.0)
    s1 = weekly_pivot - (weekly_range * 1.0)
    r2 = weekly_pivot + (weekly_range * 2.0)
    s2 = weekly_pivot - (weekly_range * 2.0)
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 60-period EMA for trend filter (10 days of 6h data)
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (00-23 UTC for 6h - less restrictive)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 0) & (hours <= 23)  # Trade all hours for 6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(atr_aligned[i]) or
            np.isnan(ema_60[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: trade all hours for 6h
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA60
        uptrend = close[i] > ema_60[i]
        downtrend = close[i] < ema_60[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter_low = atr_aligned[i] > 0
        
        # Breakout conditions: price breaks weekly S2/R2 with volume and trend
        long_breakout = close[i] > r2_aligned[i]
        short_breakout = close[i] < s2_aligned[i]
        
        # Fade conditions: price touches weekly S1/R1 with reversal signals
        long_fade = (close[i] <= s1_aligned[i] * 1.005) and (close[i] > s1_aligned[i] * 0.995)  # Near S1
        short_fade = (close[i] >= r1_aligned[i] * 0.995) and (close[i] <= r1_aligned[i] * 1.005)  # Near R1
        
        # Entry conditions
        long_entry = long_breakout and uptrend and vol_filter and vol_filter_low
        short_entry = short_breakout and downtrend and vol_filter and vol_filter_low
        
        # Fade entries (counter-trend at weekly support/resistance)
        long_fade_entry = long_fade and not uptrend and vol_filter  # Buy near S1 in downtrend
        short_fade_entry = short_fade and not downtrend and vol_filter  # Sell near R1 in uptrend
        
        # Exit conditions: return to weekly pivot or opposite extreme
        long_exit = close[i] < weekly_pivot[i] or close[i] > r2_aligned[i] * 1.02
        short_exit = close[i] > weekly_pivot[i] or close[i] < s2_aligned[i] * 0.98
        
        if (long_entry or long_fade_entry) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_entry or short_fade_entry) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_BreakoutFade_VolumeTrend"
timeframe = "6h"
leverage = 1.0