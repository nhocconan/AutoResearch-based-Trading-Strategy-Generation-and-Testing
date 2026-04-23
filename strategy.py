#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
Uses 1h timeframe for entry timing, 4h for trend direction and Camarilla levels.
Adds session filter (08-20 UTC) to reduce noise trades. Targets 15-35 trades/year per symbol.
Uses discrete position sizing (0.20) to minimize fee churn. Works in both bull and bear markets
by requiring alignment with 4h trend and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    # Camarilla calculation: based on previous 4h bar's range
    range_4h = high_4h - low_4h
    h3 = close_4h_arr + 0.275 * range_4h  # H3 level
    l3 = close_4h_arr - 0.275 * range_4h  # L3 level
    h4 = close_4h_arr + 0.550 * range_4h  # H4 level
    l4 = close_4h_arr - 0.550 * range_4h  # L4 level
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 4h EMA50 = uptrend, close < 4h EMA50 = downtrend
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: 1h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > h3_aligned[i]  # Break above H3
        breakout_down = close[i] < l3_aligned[i]  # Break below L3
        return_to_h4 = close[i] < h4_aligned[i]  # Return below H4 (exit long)
        return_to_l4 = close[i] > l4_aligned[i]  # Return above L4 (exit short)
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above H3 AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below L3 AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: return to H4/L4 or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = return_to_h4 or opposite_extreme
            elif position == -1:
                exit_signal = return_to_l4 or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0