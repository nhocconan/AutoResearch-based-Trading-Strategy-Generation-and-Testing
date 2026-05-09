#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R1_S1_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA200 for long-term trend direction
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate weekly pivot points (R1, S1) from previous week
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    R1 = 2 * pivot - prev_low
    S1 = 2 * pivot - prev_high
    
    # Align pivot levels to daily timeframe
    R1_1d = align_htf_to_ltf(prices, df_1w, R1)
    S1_1d = align_htf_to_ltf(prices, df_1w, S1)
    
    # Volume filter: above 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_1d[i]) or np.isnan(S1_1d[i]) or 
            np.isnan(ema_200_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with weekly uptrend
            if (close[i] > R1_1d[i] and 
                close[i] > ema_200_1d[i] and  # weekly uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with weekly downtrend
            elif (close[i] < S1_1d[i] and 
                  close[i] < ema_200_1d[i] and  # weekly downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below pivot (mean reversion to center)
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above pivot (mean reversion to center)
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals