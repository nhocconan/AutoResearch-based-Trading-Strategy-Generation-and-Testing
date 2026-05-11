#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla levels (R1 and S1 are the most significant for breakouts)
    # R1 = Close + (Range * 1.1/12), S1 = Close - (Range * 1.1/12)
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (using previous day's values)
    r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA for trend filter (slower to avoid whipsaws)
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=50, min_periods=50).mean().values
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.8x 20-period average (higher threshold = fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_1d_12h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above EMA (uptrend) AND volume spike
            if close[i] > r1_12h[i] and close[i] > ema_1d_12h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND below EMA (downtrend) AND volume spike
            elif close[i] < s1_12h[i] and close[i] < ema_1d_12h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR below EMA (trend change)
            if close[i] < s1_12h[i] or close[i] < ema_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR above EMA (trend change)
            if close[i] > r1_12h[i] or close[i] > ema_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals