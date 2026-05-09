#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_DailyBreakout_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Resistance 1 = (2 * PP) - Low
    r1 = (2 * pp) - weekly_low
    # Support 1 = (2 * PP) - High
    s1 = (2 * pp) - weekly_high
    
    # Align weekly pivot to daily timeframe (with 1-bar delay for completed weekly bar)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: spike above 2.0x 20-period average (approx 1 month of daily bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r1_1d[i]) or np.isnan(s1_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above weekly S1, weekly uptrend (price > EMA34), volume breakout
            if (close[i] > s1_1d[i] and 
                close[i] > ema_34_1d[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly R1, weekly downtrend (price < EMA34), volume breakdown
            elif (close[i] < r1_1d[i] and 
                  close[i] < ema_34_1d[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly S1 or trend reversal
            if close[i] < s1_1d[i] or close[i] < ema_34_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly R1 or trend reversal
            if close[i] > r1_1d[i] or close[i] > ema_34_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals