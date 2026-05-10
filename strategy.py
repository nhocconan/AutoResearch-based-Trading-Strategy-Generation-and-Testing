#!/usr/bin/env python3
# 4H_Camarilla_R3S3_Breakout_1dTrend_Volume_Filter
# Hypothesis: Trade Camarilla level (R3/S3) breakouts with daily EMA trend filter and volume confirmation.
# Works in bull/bear by following 1d trend; Camarilla levels provide structure in ranging markets.
# Volume confirmation filters false breakouts. Target: 20-30 trades/year per symbol.

name = "4H_Camarilla_R3S3_Breakout_1dTrend_Volume_Filter"
timeframe = "4h"
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
    
    # Calculate Camarilla levels from previous day
    # P = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    
    # We need previous day's OHLC for current day's Camarilla levels
    # Since we're on 4h timeframe, we'll calculate daily OHLC from 4h data
    
    # Convert 4h data to daily OHLC
    df = prices.copy()
    df['date'] = pd.to_datetime(df['open_time']).dt.date
    daily_ohlc = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day (using previous day's data)
    prev_high = daily_ohlc['high'].shift(1).values
    prev_low = daily_ohlc['low'].shift(1).values
    prev_close = daily_ohlc['close'].shift(1).values
    
    # Calculate P and range
    P = (prev_high + prev_low + prev_close) / 3
    rang = prev_high - prev_low
    
    # R3 and S3
    R3 = prev_close + rang * 1.1 / 2
    S3 = prev_close - rang * 1.1 / 2
    
    # Map daily levels back to 4h timeframe
    # Create a mapping from date to Camarilla levels
    date_to_R3 = dict(zip(daily_ohlc['date'][1:], R3[1:]))  # Skip first day (no prev data)
    date_to_S3 = dict(zip(daily_ohlc['date'][1:], S3[1:]))
    
    # Get date for each 4h bar
    dates = pd.to_datetime(df['open_time']).dt.date
    
    # Map Camarilla levels to each 4h bar
    R3_levels = np.array([date_to_R3.get(d, np.nan) for d in dates])
    S3_levels = np.array([date_to_S3.get(d, np.nan) for d in dates])
    
    # Daily trend filter (using 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_levels[i]) or np.isnan(S3_levels[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(daily_uptrend_aligned[i]) or 
            np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / volume_ma[i] if volume_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R3 + volume
            if daily_up and close[i] > R3_levels[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + price breaks below S3 + volume
            elif daily_down and close[i] < S3_levels[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend changes or price moves back below R3
            if not daily_up or close[i] < R3_levels[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend changes or price moves back above S3
            if not daily_down or close[i] > S3_levels[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals