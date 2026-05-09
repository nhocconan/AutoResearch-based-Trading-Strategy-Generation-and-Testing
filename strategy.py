#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (using previous day's range)
    range_1d = prev_high_1d - prev_low_1d
    camarilla_h4 = prev_close_1d + 1.5 * range_1d  # Resistance level 4
    camarilla_l4 = prev_close_1d - 1.5 * range_1d  # Support level 4
    camarilla_h3 = prev_close_1d + 1.125 * range_1d  # Resistance level 3 (exit)
    camarilla_l3 = prev_close_1d - 1.125 * range_1d  # Support level 3 (exit)
    
    # Align Camarilla levels to 12h
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: above 1.3x 12-period average (6 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 12  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.3 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long entry: price breaks above Camarilla H4 with daily uptrend
            if (close[i] > camarilla_h4_12h[i] and 
                close[i] > ema_34_12h[i] and  # daily uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Camarilla L4 with daily downtrend
            elif (close[i] < camarilla_l4_12h[i] and 
                  close[i] < ema_34_12h[i] and  # daily downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla H3 (midpoint)
            if not np.isnan(camarilla_h3_12h[i]) and close[i] < camarilla_h3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price rises back above Camarilla L3 (midpoint)
            if not np.isnan(camarilla_l3_12h[i]) and close[i] > camarilla_l3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals