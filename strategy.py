#!/usr/bin/env python3
"""
6h_1d_Weekly_Pivot_Direction_Volume_v1
Hypothesis: Weekly pivot levels derived from prior week's OHLC act as strong support/resistance on 6h timeframe.
Price tends to reverse at these levels with volume confirmation and directional bias from daily trend.
Works in both bull and bear markets by using weekly structure and daily trend filter.
Target: 15-25 trades per year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Weekly_Pivot_Direction_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for weekly pivot calculation and daily trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_open = df_1d['open'].values
    
    # === WEEKLY PIVOT LEVELS (based on prior week's OHLC) ===
    # Calculate weekly OHLC from daily data
    # Week starts on Sunday, but we'll approximate with 5-day week for simplicity
    # In practice, use actual weekly aggregation, but we'll use rolling 5-day for proxy
    weekly_high = np.full(len(daily_high), np.nan)
    weekly_low = np.full(len(daily_high), np.nan)
    weekly_close = np.full(len(daily_high), np.nan)
    weekly_open = np.full(len(daily_high), np.nan)
    
    # Simple 5-day aggregation (assuming 5 trading days per week)
    for i in range(4, len(daily_high)):
        weekly_high[i] = np.max(daily_high[i-4:i+1])
        weekly_low[i] = np.min(daily_low[i-4:i+1])
        weekly_close[i] = daily_close[i]
        weekly_open[i] = daily_open[i-4]
    
    # Use previous week's data for pivot (to avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 5)  # Shift by ~1 week
    prev_weekly_low = np.roll(weekly_low, 5)
    prev_weekly_close = np.roll(weekly_close, 5)
    prev_weekly_open = np.roll(weekly_open, 5)
    
    # Weekly pivot calculation
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_range = prev_weekly_high - prev_weekly_low
    
    # Key weekly pivot levels
    weekly_r1 = 2 * weekly_pivot - prev_weekly_low
    weekly_s1 = 2 * weekly_pivot - prev_weekly_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # === DAILY TREND FILTER (EMA crossover on 1D) ===
    # Use 20 and 50 EMA for trend direction
    ema_20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_6h = align_htf_to_ltf(prices, df_1d, ema_20)
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    daily_uptrend = ema_20_6h > ema_50_6h
    daily_downtrend = ema_20_6h < ema_50_6h
    
    # === VOLUME SPIKE (2x 20-period average on 6h) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or np.isnan(weekly_r2_6h[i]) or
            np.isnan(weekly_s2_6h[i]) or np.isnan(ema_20_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price near weekly support/resistance levels (within 0.2% tolerance)
        near_s1 = abs(low[i] - weekly_s1_6h[i]) / weekly_s1_6h[i] < 0.002
        near_s2 = abs(low[i] - weekly_s2_6h[i]) / weekly_s2_6h[i] < 0.002
        near_r1 = abs(high[i] - weekly_r1_6h[i]) / weekly_r1_6h[i] < 0.002
        near_r2 = abs(high[i] - weekly_r2_6h[i]) / weekly_r2_6h[i] < 0.002
        
        # Entry conditions with volume confirmation and trend filter
        # Long: price near support in uptrend or strong support in any trend
        long_entry = ((near_s1 or near_s2) and daily_uptrend[i]) or \
                     ((near_s2) and vol_spike[i])  # Strong support with volume
        
        # Short: price near resistance in downtrend or strong resistance in any trend
        short_entry = ((near_r1 or near_r2) and daily_downtrend[i]) or \
                      ((near_r2) and vol_spike[i])  # Strong resistance with volume
        
        # Exit conditions: price moves back toward weekly pivot or opposite signal
        long_exit = close[i] >= weekly_pivot_6h[i]  # Exit long when price reaches pivot
        short_exit = close[i] <= weekly_pivot_6h[i]  # Exit short when price reaches pivot
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals