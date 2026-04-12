#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_volume_v1
# Use 1d Camarilla levels for directional bias, 4h for trend confirmation, and 1h for precise entry timing.
# Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (15-30/year) to minimize fee drag in challenging 1h timeframe.
# Works in bull markets by buying dips to L3/L4 in uptrend, and in bear markets by selling rallies to H3/H4.
name = "1h_4h_1d_camarilla_volume_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_level = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 4h data for trend confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA(20) for trend
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 2.0 * 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready or outside session
        if (np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or 
            np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or 
            np.isnan(ema_4h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Skip if volume confirmation fails
        if not vol_confirm[i]:
            # Hold current position if volume fails
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: price near support in uptrend
        if (close[i] <= l3_level[i] and 
            ema_4h_aligned[i] > close[i] and  # slight dip below 4h EMA in uptrend
            position != 1):
            position = 1
            signals[i] = 0.20
        # Short conditions: price near resistance in downtrend
        elif (close[i] >= h3_level[i] and 
              ema_4h_aligned[i] < close[i] and  # slight rally above 4h EMA in downtrend
              position != -1):
            position = -1
            signals[i] = -0.20
        # Exit conditions: reverse signal or extreme levels
        elif (close[i] >= h4_level[i] and position == 1) or \
             (close[i] <= l4_level[i] and position == -1) or \
             (close[i] >= ema_4h_aligned[i] and position == 1 and ema_4h_aligned[i] > close[i-1]) or \
             (close[i] <= ema_4h_aligned[i] and position == -1 and ema_4h_aligned[i] < close[i-1]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals