#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load 1d data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (using previous day's data)
    # Camarilla: Pivot = (H+L+C)/3
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use previous day's OHLC to avoid look-ahead
    shift_high_1d = np.roll(high_1d, 1)
    shift_low_1d = np.roll(low_1d, 1)
    shift_close_1d = np.roll(close_1d, 1)
    # First day: use available data (will be NaN until we have enough)
    shift_high_1d[0] = high_1d[0]
    shift_low_1d[0] = low_1d[0]
    shift_close_1d[0] = close_1d[0]
    
    camarilla_pivot = (shift_high_1d + shift_low_1d + shift_close_1d) / 3
    camarilla_range = shift_high_1d - shift_low_1d
    camarilla_r1 = camarilla_pivot + camarilla_range * 1.1 / 12
    camarilla_s1 = camarilla_pivot - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above 1d EMA34 + volume filter
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 + below 1d EMA34 + volume filter
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R1
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals