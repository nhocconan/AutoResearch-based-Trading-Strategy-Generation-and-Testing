#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Swing_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE for weekly pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly pivot calculation (using previous week's OHLC)
    # We need to resample daily to weekly to get proper weekly OHLC
    # But since we can't use resample, we'll approximate using last 5 days
    # For weekly pivot, we use: (weekly_high + weekly_low + weekly_close) / 3
    # We'll calculate weekly OHLC from daily data
    
    # Calculate weekly high, low, close using rolling window of 5 days
    # This approximates weekly data (5 trading days)
    week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (week_high + week_low + week_close) / 3
    weekly_range = week_high - week_low
    
    # Weekly support/resistance levels (similar to Camarilla but simpler)
    # R1 = pivot + (range * 1.0)
    # S1 = pivot - (range * 1.0)
    weekly_r1 = weekly_pivot + weekly_range
    weekly_s1 = weekly_pivot - weekly_range
    
    # Align weekly levels to 6h timeframe
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Daily EMA21 for trend filter
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or 
            np.isnan(ema_21_6h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(weekly_pivot_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price bounces off weekly S1 in daily uptrend with volume
            if low[i] <= weekly_s1_6h[i] and close[i] > weekly_s1_6h[i] and ema_21_6h[i] > ema_21_6h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price rejects at weekly R1 in daily downtrend with volume
            elif high[i] >= weekly_r1_6h[i] and close[i] < weekly_r1_6h[i] and ema_21_6h[i] < ema_21_6h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches weekly pivot or trend reverses
            if close[i] >= weekly_pivot_6h[i] or ema_21_6h[i] < ema_21_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches weekly pivot or trend reverses
            if close[i] <= weekly_pivot_6h[i] or ema_21_6h[i] > ema_21_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot swing strategy for 6m timeframe
# - Uses weekly pivot points (R1/S1) derived from prior 5-day weekly OHLC
# - Long when price bounces off weekly S1 in daily uptrend (EMA21 rising) with volume confirmation
# - Short when price rejects at weekly R1 in daily downtrend (EMA21 falling) with volume confirmation
# - Exit when price returns to weekly pivot or daily trend reverses
# - Works in both bull (bounces at support in uptrend) and bear (rejections at resistance in downtrend)
# - Volume filter (1.5x average) reduces false signals
# - Position size 0.25 targets ~20-40 trades/year to avoid excessive fee drag
# - Weekly pivot provides institutional reference points that work across market regimes
# - Uses 1d timeframe for trend filter and 6h for execution timing
# - Designed to capture mean reversion at key weekly levels with trend alignment