#!/usr/bin/env python3
name = "4h_Camarilla_R2_S2_Breakout_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 1d data for Camarilla pivots (from previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla R2 and S2 levels
    camarilla_r2 = close_1d + (range_1d * 1.1 / 6)
    camarilla_s2 = close_1d - (range_1d * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe (using previous 1d bar's values)
    r2_4h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume filter: current volume > 1.6x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 AND above 1d EMA34 (uptrend) AND volume surge
            if close[i] > r2_4h[i] and close[i] > ema_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 AND below 1d EMA34 (downtrend) AND volume surge
            elif close[i] < s2_4h[i] and close[i] < ema_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S2 OR below 1d EMA34 (trend change)
            if close[i] < s2_4h[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R2 OR above 1d EMA34 (trend change)
            if close[i] > r2_4h[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals