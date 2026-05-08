# 6H Weekly Pivot + Daily Trend Breakout
# Combines weekly pivot points (calculated from weekly OHLC) with daily trend filter
# Long: price breaks above weekly pivot with daily uptrend
# Short: price breaks below weekly pivot with daily downtrend
# Weekly pivot provides institutional levels, daily trend ensures directional alignment
# Target: 15-30 trades/year to minimize fee drag on 6h timeframe
# Works in bull/bear by using daily trend filter to avoid counter-trend trades

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyTrend_Breakout"
timeframe = "6h"
leverage = 1.0

def calculate_weekly_pivot(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2*P - H, R1 = 2*P - L
    Returns arrays aligned to weekly bar closes
    """
    typical_price = (high + low + close) / 3.0
    pivot = typical_price
    s1 = 2 * pivot - high
    r1 = 2 * pivot - low
    return pivot, s1, r1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    n_weekly = len(weekly_close)
    weekly_pivot = np.full(n_weekly, np.nan)
    weekly_s1 = np.full(n_weekly, np.nan)
    weekly_r1 = np.full(n_weekly, np.nan)
    
    for i in range(1, n_weekly):
        H = weekly_high[i-1]
        L = weekly_low[i-1]
        C = weekly_close[i-1]
        weekly_pivot[i], weekly_s1[i], weekly_r1[i] = calculate_weekly_pivot(H, L, C)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    # Daily EMA20 for trend filter
    ema_20_daily = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(ema_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly pivot with daily uptrend
            long_cond = (close[i] > weekly_pivot_aligned[i] and 
                        ema_20_daily_aligned[i] > ema_20_daily_aligned[i-1])
            
            # Short: price breaks below weekly pivot with daily downtrend
            short_cond = (close[i] < weekly_pivot_aligned[i] and 
                         ema_20_daily_aligned[i] < ema_20_daily_aligned[i-1])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 (strong support)
            if close[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly R1 (strong resistance)
            if close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyTrend_Breakout"
timeframe = "6h"
leverage = 1.0

def calculate_weekly_pivot(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2*P - H, R1 = 2*P - L
    Returns arrays aligned to weekly bar closes
    """
    typical_price = (high + low + close) / 3.0
    pivot = typical_price
    s1 = 2 * pivot - high
    r1 = 2 * pivot - low
    return pivot, s1, r1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    n_weekly = len(weekly_close)
    weekly_pivot = np.full(n_weekly, np.nan)
    weekly_s1 = np.full(n_weekly, np.nan)
    weekly_r1 = np.full(n_weekly, np.nan)
    
    for i in range(1, n_weekly):
        H = weekly_high[i-1]
        L = weekly_low[i-1]
        C = weekly_close[i-1]
        weekly_pivot[i], weekly_s1[i], weekly_r1[i] = calculate_weekly_pivot(H, L, C)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    # Daily EMA20 for trend filter
    ema_20_daily = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(ema_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly pivot with daily uptrend
            long_cond = (close[i] > weekly_pivot_aligned[i] and 
                        ema_20_daily_aligned[i] > ema_20_daily_aligned[i-1])
            
            # Short: price breaks below weekly pivot with daily downtrend
            short_cond = (close[i] < weekly_pivot_aligned[i] and 
                         ema_20_daily_aligned[i] < ema_20_daily_aligned[i-1])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 (strong support)
            if close[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly R1 (strong resistance)
            if close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals