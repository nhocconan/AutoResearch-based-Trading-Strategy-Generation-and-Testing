#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_WeeklyTrend
Hypothesis: KAMA adapts to market efficiency, filtering noise in chop and capturing trends.
Combines KAMA direction with weekly trend filter and volume confirmation to reduce whipsaw.
Works in bull markets by catching trends and in bear markets by avoiding false signals during consolidation.
Designed for low trade frequency (<10 trades/year) to minimize fee drag.
"""

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
    
    # Calculate KAMA (adaptive moving average) on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array alignment: volatility needs to be same length as change
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+10]))) if i+10 <= len(close) else np.nan 
                          for i in range(len(close))])
    # Simpler approach: calculate ER using rolling
    close_series = pd.Series(close)
    change = np.abs(close_series.diff(10))
    volatility = close_series.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    # Handle NaN in sc
    sc = sc.fillna(0)
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i-1]) or sc[i-1] == 0:
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1])
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: KAMA direction aligned with weekly trend + volume
        long_entry = vol_confirm and kama_bullish and weekly_uptrend
        short_entry = vol_confirm and kama_bearish and weekly_downtrend
        
        # Exit logic: opposite KAMA cross or weekly trend change
        long_exit = (not kama_bullish) or (not weekly_uptrend)
        short_exit = (not kama_bearish) or (not weekly_downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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

name = "1d_KAMA_Trend_Filter_WeeklyTrend"
timeframe = "1d"
leverage = 1.0