#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Filter_v1
Hypothesis: Use 1h Camarilla pivot breakout (R1/S1) with 4h trend filter (EMA50) and session filter (08-20 UTC). 
Camarilla levels provide high-probability reversal/breakout zones. 4h EMA50 ensures trading with higher timeframe trend.
Session filter reduces noise during low-volume hours. Designed for 1h timeframe to achieve 15-37 trades/year.
Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes in ranging conditions).
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Filter_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivots using previous day's OHLC
    # We need to group by day to get previous day's OHLC
    # Create date column for grouping
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    # Arrays to store Camarilla levels for each bar
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    
    # Calculate for each day
    for i, date in enumerate(unique_dates):
        # Get indices for this day
        day_mask = (dates == date)
        if not np.any(day_mask):
            continue
            
        # Get previous day's data (if exists)
        if i == 0:
            # First day, no previous day
            continue
            
        prev_date = unique_dates[i-1]
        prev_mask = (dates == prev_date)
        
        if not np.any(prev_mask):
            continue
            
        # Previous day's OHLC
        prev_high = np.max(high[prev_mask])
        prev_low = np.min(low[prev_mask])
        prev_close = close[prev_mask][-1]  # Last close of previous day
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            continue
            
        # Camarilla R1 and S1
        R1_val = prev_close + (range_val * 1.1 / 12)
        S1_val = prev_close - (range_val * 1.1 / 12)
        
        # Apply to current day's bars
        R1[day_mask] = R1_val
        S1[day_mask] = S1_val
    
    # Session filter: 08-20 UTC
    hours = pd.to_datetime(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data is not available
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with 4h uptrend (price > EMA50)
            if close[i] > R1[i] and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 with 4h downtrend (price < EMA50)
            elif close[i] < S1[i] and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or 4h trend turns down
            if close[i] < S1[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or 4h trend turns up
            if close[i] > R1[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals