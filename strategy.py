#!/usr/bin/env python3
"""
Hypothesis: 1d 1-week Supertrend with 1d RSI filter and volume confirmation.
Uses 1-week Supertrend (ATR=10, multiplier=3) for trend direction, 1d RSI(14) for momentum 
confirmation (long when RSI > 50, short when RSI < 50), and 1d volume spike (volume > 1.5x 
20-period average) to confirm momentum. Long when 1-week Supertrend is bullish, RSI > 50, 
and volume spike. Short when 1-week Supertrend is bearish, RSI < 50, and volume spike.
Designed to work in both bull and bear markets by following the higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
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
    
    # Get 1d data for RSI and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50], rsi])  # pad first value
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    supertrend = np.zeros_like(close_1w)
    trend = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            trend[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if trend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if trend[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Supertrend direction: 1 for uptrend, -1 for downtrend
    supertrend_direction = trend
    
    # Align 1d indicators
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Align 1w Supertrend direction
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, supertrend_direction.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(supertrend_direction_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Supertrend direction + RSI > 50/<50 + volume spike
        supertrend_up = supertrend_direction_aligned[i] > 0
        supertrend_down = supertrend_direction_aligned[i] < 0
        rsi_above = rsi_aligned[i] > 50
        rsi_below = rsi_aligned[i] < 50
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        
        long_entry = supertrend_up and rsi_above and vol_confirm
        short_entry = supertrend_down and rsi_below and vol_confirm
        
        # Exit when Supertrend reverses
        exit_long = position == 1 and supertrend_down
        exit_short = position == -1 and supertrend_up
        
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

name = "1d_1w_supertrend_rsi_volume"
timeframe = "1d"
leverage = 1.0