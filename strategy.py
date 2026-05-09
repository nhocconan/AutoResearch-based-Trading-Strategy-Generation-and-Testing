#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_Pivot_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Previous 4h bar's OHLC for Camarilla calculation
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    # Calculate Camarilla levels (using previous 4h bar's range)
    range_4h = prev_high_4h - prev_low_4h
    camarilla_h3 = prev_close_4h + 1.125 * range_4h  # Resistance level 3
    camarilla_l3 = prev_close_4h - 1.125 * range_4h  # Support level 3
    
    # Align Camarilla levels to 1h
    camarilla_h3_1h = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_1h = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1h = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume filter: above 1.5x 12-period average (12*1h = 12h)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 12  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_1h[i]) or np.isnan(camarilla_l3_1h[i]) or 
            np.isnan(ema_34_1h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long entry: price breaks above Camarilla H3 with 4h uptrend
            if (close[i] > camarilla_h3_1h[i] and 
                close[i] > ema_34_1h[i] and  # 4h uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla L3 with 4h downtrend
            elif (close[i] < camarilla_l3_1h[i] and 
                  close[i] < ema_34_1h[i] and  # 4h downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla H2 (lower resistance)
            camarilla_h2 = prev_close_4h + 0.75 * range_4h  # Resistance level 2
            camarilla_h2_1h = align_htf_to_ltf(prices, df_4h, camarilla_h2)
            if not np.isnan(camarilla_h2_1h[i]) and close[i] < camarilla_h2_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above Camarilla L2 (higher support)
            camarilla_l2 = prev_close_4h - 0.75 * range_4h  # Support level 2
            camarilla_l2_1h = align_htf_to_ltf(prices, df_4h, camarilla_l2)
            if not np.isnan(camarilla_l2_1h[i]) and close[i] > camarilla_l2_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals