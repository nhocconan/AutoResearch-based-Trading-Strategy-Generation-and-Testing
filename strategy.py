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
    
    # Get 1d data for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day OHLC)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    
    # Align daily pivots to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily trend: price above/below daily EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly trend: price above/below weekly EMA(20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: volume > 1.5 x 20-period average (daily)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need pivots (1), daily EMA (34), weekly EMA (20), volume MA (20)
    start_idx = max(1, 34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Daily trend filter
        bullish_daily = price > ema_34_1d_aligned[i]
        bearish_daily = price < ema_34_1d_aligned[i]
        
        # Weekly trend filter
        bullish_weekly = price > ema_20_1w_aligned[i]
        bearish_weekly = price < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price crosses above S1 with volume and bullish daily AND weekly trend
            if price > s1_aligned[i] and vol_filter and bullish_daily and bullish_weekly:
                signals[i] = size
                position = 1
            # Short: price crosses below R1 with volume and bearish daily AND weekly trend
            elif price < r1_aligned[i] and vol_filter and bearish_daily and bearish_weekly:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below pivot or daily trend turns bearish
            if price < pivot_aligned[i] or not bullish_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above pivot or daily trend turns bullish
            if price > pivot_aligned[i] or not bearish_daily:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Pivot_S1R1_DailyWeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0