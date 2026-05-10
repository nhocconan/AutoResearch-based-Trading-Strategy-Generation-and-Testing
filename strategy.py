#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Breakout above/below Camarilla R1/S1 levels on 12h, filtered by 1w EMA20 trend and volume confirmation (>1.5x average). Uses ATR-based stoploss. Designed for 15-30 trades/year to avoid fee drag. Works in bull/bear via trend filter.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Get 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily high/low for Camarilla levels (previous day)
    daily_high = np.full(n, np.nan)
    daily_low = np.full(n, np.nan)
    daily_close = np.full(n, np.nan)
    
    # Group by date to get daily OHLC
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    
    # Initialize daily arrays
    daily_high_arr = np.full(len(unique_dates), np.nan)
    daily_low_arr = np.full(len(unique_dates), np.nan)
    daily_close_arr = np.full(len(unique_dates), np.nan)
    
    # Calculate daily OHLC
    for i in range(n):
        date_idx = date_to_idx[dates[i]]
        if np.isnan(daily_high_arr[date_idx]) or high[i] > daily_high_arr[date_idx]:
            daily_high_arr[date_idx] = high[i]
        if np.isnan(daily_low_arr[date_idx]) or low[i] < daily_low_arr[date_idx]:
            daily_low_arr[date_idx] = low[i]
        daily_close_arr[date_idx] = close[i]
    
    # Map daily values back to 12h bars
    for i in range(n):
        date_idx = date_to_idx[dates[i]]
        daily_high[i] = daily_high_arr[date_idx]
        daily_low[i] = daily_low_arr[date_idx]
        daily_close[i] = daily_close_arr[date_idx]
    
    # Calculate Camarilla levels (based on previous day's range)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            continue  # No previous day for first bar
        prev_date_idx = date_to_idx[dates[i]] - 1
        if prev_date_idx < 0:
            continue  # No previous day data
        if np.isnan(daily_high_arr[prev_date_idx]) or np.isnan(daily_low_arr[prev_date_idx]) or np.isnan(daily_close_arr[prev_date_idx]):
            continue
        rng = daily_high_arr[prev_date_idx] - daily_low_arr[prev_date_idx]
        camarilla_r1[i] = daily_close_arr[prev_date_idx] + rng * 1.1 / 12
        camarilla_s1[i] = daily_close_arr[prev_date_idx] - rng * 1.1 / 12
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1w EMA20 trend
            if close[i] > ema_20_1w_aligned[i]:  # Uptrend
                # Long: Breakout above Camarilla R1 with volume confirmation
                if close[i] > camarilla_r1[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Breakout below Camarilla S1 with volume confirmation
                if close[i] < camarilla_s1[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA20 or stoploss hit
            if close[i] < ema_20_1w_aligned[i] or (i > 0 and low[i] < camarilla_s1[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA20 or stoploss hit
            if close[i] > ema_20_1w_aligned[i] or (i > 0 and high[i] > camarilla_r1[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals