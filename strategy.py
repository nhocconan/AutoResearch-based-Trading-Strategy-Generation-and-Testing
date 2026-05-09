#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Previous 1d close for trend filter
    prev_close_1d = df_1d['close'].shift(1).values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous 12h bar's OHLC for Camarilla calculation
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    
    # Calculate Camarilla levels (using previous 12h bar's range)
    range_12h = prev_high_12h - prev_low_12h
    camarilla_h4 = prev_close_12h + 1.5 * range_12h  # Resistance level 4
    camarilla_l4 = prev_close_12h - 1.5 * range_12h  # Support level 4
    camarilla_h3 = prev_close_12h + 1.125 * range_12h  # Resistance level 3
    camarilla_l3 = prev_close_12h - 1.125 * range_12h  # Support level 3
    
    # Align Camarilla levels to 12h
    camarilla_h4_12h = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    camarilla_h3_12h = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_12h = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Volume filter: above 2x 24-period average (24*12h = 12d)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Strong volume confirmation
        
        # Pre-compute hour for session filter
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price breaks above Camarilla H4 with 1d uptrend
            if (close[i] > camarilla_h4_12h[i] and 
                close[i] > ema_50_12h[i] and  # 1d uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L4 with 1d downtrend
            elif (close[i] < camarilla_l4_12h[i] and 
                  close[i] < ema_50_12h[i] and  # 1d downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla H3 (strong resistance)
            if not np.isnan(camarilla_h3_12h[i]) and close[i] < camarilla_h3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Camarilla L3 (strong support)
            if not np.isnan(camarilla_l3_12h[i]) and close[i] > camarilla_l3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals