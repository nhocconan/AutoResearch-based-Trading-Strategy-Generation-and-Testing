#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_12hTrend_Volume"
timeframe = "6h"
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
    
    # Calculate weekly pivot points from Monday's OHLC (00:00 UTC Monday)
    weekly_pivot = np.full(n, np.nan)
    weekly_R1 = np.full(n, np.nan)
    weekly_S1 = np.full(n, np.nan)
    weekly_R2 = np.full(n, np.nan)
    weekly_S2 = np.full(n, np.nan)
    
    # Get Monday 00:00 UTC timestamps for each week
    open_time = pd.to_datetime(prices['open_time'])
    # Find start of week (Monday 00:00 UTC)
    week_start = open_time - pd.to_timedelta(open_time.dt.weekday, unit='D') 
    week_start = week_start.dt.normalize()  # Set to 00:00:00
    
    # Group by week start to get weekly OHLC
    weekly_data = {}
    for i in range(n):
        ws = week_start.iloc[i]
        if ws not in weekly_data:
            weekly_data[ws] = {'high': high[i], 'low': low[i], 'close': close[i], 'open': open_time[i]}
        else:
            weekly_data[ws]['high'] = max(weekly_data[ws]['high'], high[i])
            weekly_data[ws]['low'] = min(weekly_data[ws]['low'], low[i])
            weekly_data[ws]['close'] = close[i]
    
    # Calculate pivot points for each week
    for ws, data in weekly_data.items():
        # Use previous week's data for current week's pivot (avoid look-ahead)
        pass  # Will handle in loop below
    
    # Simpler approach: calculate pivot from previous week's OHLC
    weekly_high = np.full(n, np.nan)
    weekly_low = np.full(n, np.nan)
    weekly_close = np.full(n, np.nan)
    
    for i in range(n):
        ws = week_start.iloc[i]
        if ws in weekly_data:
            weekly_high[i] = weekly_data[ws]['high']
            weekly_low[i] = weekly_data[ws]['low']
            weekly_close[i] = weekly_data[ws]['close']
    
    # Shift by one week to use previous week's data
    weekly_high = np.roll(weekly_high, 1)
    weekly_low = np.roll(weekly_low, 1)
    weekly_close = np.roll(weekly_close, 1)
    # Set first week's values to NaN (no prior week)
    weekly_high[:7*24//6] = np.nan  # Approximate first week
    weekly_low[:7*24//6] = np.nan
    weekly_close[:7*24//6] = np.nan
    
    # Calculate pivot points using previous week's OHLC
    for i in range(n):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            pp = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3
            weekly_pivot[i] = pp
            weekly_R1[i] = 2 * pp - weekly_low[i]
            weekly_S1[i] = 2 * pp - weekly_high[i]
            weekly_R2[i] = pp + (weekly_high[i] - weekly_low[i])
            weekly_S2[i] = pp - (weekly_high[i] - weekly_low[i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.8 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_R2[i]) or np.isnan(weekly_S2[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition
        vol_condition = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above weekly R2 with 12h uptrend and volume
            if close[i] > weekly_R2[i] and close[i] > ema50_12h_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S2 with 12h downtrend and volume
            elif close[i] < weekly_S2[i] and close[i] < ema50_12h_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters weekly pivot range (below R2) or trend reversal
            if close[i] < weekly_R2[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters weekly pivot range (above S2) or trend reversal
            if close[i] > weekly_S2[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals