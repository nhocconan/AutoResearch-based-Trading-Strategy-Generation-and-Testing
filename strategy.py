#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_Pivot_4hTrend_Volume_v4"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and daily context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Previous 1d close for trend filter
    prev_close_1d = df_1d['close'].shift(1).values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous 4h bar's OHLC for Camarilla calculation
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    # Calculate Camarilla levels (using previous 4h bar's range)
    range_4h = prev_high_4h - prev_low_4h
    camarilla_h4 = prev_close_4h + 1.5 * range_4h  # Resistance level 4
    camarilla_l4 = prev_close_4h - 1.5 * range_4h  # Support level 4
    
    # Align Camarilla levels to 1h
    camarilla_h4_1h = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_1h = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Volume filter: above 2x 24-period average (24*1h = 24h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_1h[i]) or np.isnan(camarilla_l4_1h[i]) or 
            np.isnan(ema_50_1h[i]) or np.isnan(vol_ma[i])):
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
            if (close[i] > camarilla_h4_1h[i] and 
                close[i] > ema_50_1h[i] and  # 1d uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L4 with 1d downtrend
            elif (close[i] < camarilla_l4_1h[i] and 
                  close[i] < ema_50_1h[i] and  # 1d downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla H3 (strong resistance)
            camarilla_h3 = prev_close_4h + 1.125 * range_4h  # Resistance level 3
            camarilla_h3_1h = align_htf_to_ltf(prices, df_4h, camarilla_h3)
            if not np.isnan(camarilla_h3_1h[i]) and close[i] < camarilla_h3_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above Camarilla L3 (strong support)
            camarilla_l3 = prev_close_4h - 1.125 * range_4h  # Support level 3
            camarilla_l3_1h = align_htf_to_ltf(prices, df_4h, camarilla_l3)
            if not np.isnan(camarilla_l3_1h[i]) and close[i] > camarilla_l3_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals