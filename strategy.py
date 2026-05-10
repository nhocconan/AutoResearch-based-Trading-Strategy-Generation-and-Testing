#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_With_Trend
# Hypothesis: Weekly pivot points (from weekly high/low/close) act as strong support/resistance.
# In uptrends (price > daily EMA50), we go long on breakout above weekly R1 with volume confirmation.
# In downtrends (price < daily EMA50), we go short on breakdown below weekly S1 with volume confirmation.
# Uses weekly pivot levels calculated from prior week's OHLC to avoid look-ahead.
# Designed for low trade frequency (15-30/year) to minimize fee flood.

name = "6h_Weekly_Pivot_Breakout_With_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation (use prior week's OHLC)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Get daily EMA50 for trend filter
    df_daily = get_htf_data(prices, '1d')
    daily_close = df_daily['close'].values
    ema_50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume confirmation (24-period average on 6h = ~6 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24) + 5  # need enough history
    
    for i in range(start_idx, n):
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or \
           np.isnan(weekly_s1_aligned[i]) or np.isnan(ema_50_daily_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume, in uptrend (price > daily EMA50)
            if close[i] > weekly_r1_aligned[i] and volume_confirm and close[i] > ema_50_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume, in downtrend (price < daily EMA50)
            elif close[i] < weekly_s1_aligned[i] and volume_confirm and close[i] < ema_50_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to weekly pivot or breaks below weekly S1
            if close[i] <= weekly_pivot_aligned[i] or close[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to weekly pivot or breaks above weekly R1
            if close[i] >= weekly_pivot_aligned[i] or close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals