#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC (for Camarilla calculation)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels R1 and S1 (inner bounds)
    camarilla_pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_range_1d = prev_high_1d - prev_low_1d
    camarilla_r1_1d = camarilla_pivot_1d + camarilla_range_1d * 1.1 / 12
    camarilla_s1_1d = camarilla_pivot_1d - camarilla_range_1d * 1.1 / 12
    
    # Align Camarilla levels to 1d (same timeframe, but need alignment for proper indexing)
    camarilla_pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: above 2.0x 20-period average (20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_1d_aligned[i]) or np.isnan(camarilla_s1_1d_aligned[i]) or 
            np.isnan(ema_34_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        if position == 0:
            # Long breakout: price breaks above camarilla R1 with weekly uptrend
            if (close[i] > camarilla_r1_1d_aligned[i] and 
                close[i] > ema_34_1d[i] and  # weekly uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below camarilla S1 with weekly downtrend
            elif (close[i] < camarilla_s1_1d_aligned[i] and 
                  close[i] < ema_34_1d[i] and  # weekly downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals