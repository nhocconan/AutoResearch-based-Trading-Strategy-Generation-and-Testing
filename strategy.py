#!/usr/bin/env python3
"""
Hypothesis: 4h 1-day Camarilla pivot with 1-hour volume confirmation and 4-hour trend filter.
Uses 1-day Camarilla levels for mean reversion at extremes, 1-hour volume spike to confirm 
momentum at pivot touches, and 4-hour EMA trend filter to avoid counter-trend trades. 
Long when price touches Camarilla L4 with volume spike and above 4h EMA50. 
Short when price touches Camarilla H4 with volume spike and below 4h EMA50.
Target: 80-150 total trades over 4 years (20-38/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_h4 = typical_price + (range_1d * 1.1 / 2)
    camarilla_l4 = typical_price - (range_1d * 1.1 / 2)
    camarilla_h3 = typical_price + (range_1d * 1.1 / 4)
    camarilla_l3 = typical_price - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 1h data for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h volume spike (volume > 2x 24-period average)
    vol_ma_24 = pd.Series(volume_1h).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume_1h > (vol_ma_24 * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1h, vol_spike.astype(float))
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla touch + volume spike + trend filter
        touch_l4 = low[i] <= camarilla_l4_aligned[i]  # Touch or penetrate L4
        touch_h4 = high[i] >= camarilla_h4_aligned[i]  # Touch or penetrate H4
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        uptrend = close[i] > ema_50_aligned[i]    # Above EMA50 = uptrend
        downtrend = close[i] < ema_50_aligned[i]  # Below EMA50 = downtrend
        
        long_entry = touch_l4 and vol_confirm and uptrend
        short_entry = touch_h4 and vol_confirm and downtrend
        
        # Exit when price returns to opposite Camarilla H3/L3 level
        exit_long = position == 1 and high[i] >= camarilla_h3_aligned[i]
        exit_short = position == -1 and low[i] <= camarilla_l3_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_volume_trend"
timeframe = "4h"
leverage = 1.0