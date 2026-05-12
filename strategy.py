# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "1h_4h1d_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Daily data for trend filter and volume filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels using previous 4h bar (to avoid look-ahead)
    shift_high_4h = np.roll(high_4h, 1)
    shift_low_4h = np.roll(low_4h, 1)
    shift_close_4h = np.roll(close_4h, 1)
    # First bar: use current values to avoid look-ahead on first bar
    shift_high_4h[0] = high_4h[0]
    shift_low_4h[0] = low_4h[0]
    shift_close_4h[0] = close_4h[0]
    
    camarilla_pivot = (shift_high_4h + shift_low_4h + shift_close_4h) / 3
    camarilla_range = shift_high_4h - shift_low_4h
    camarilla_r1 = camarilla_pivot + camarilla_range * 1.1 / 12
    camarilla_s1 = camarilla_pivot - camarilla_range * 1.1 / 12
    camarilla_r2 = camarilla_pivot + camarilla_range * 1.1 / 6
    camarilla_s2 = camarilla_pivot - camarilla_range * 1.1 / 6
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s2)
    
    # Daily EMA 50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.8x 24-period average (1 day of 1h data)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_r2_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_filter[i]) or 
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade during session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above camarilla R1 + above daily EMA50 + volume filter
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema_50_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below camarilla S1 + below daily EMA50 + volume filter
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_50_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below camarilla S1 or below daily EMA50
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above camarilla R1 or above daily EMA50
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals