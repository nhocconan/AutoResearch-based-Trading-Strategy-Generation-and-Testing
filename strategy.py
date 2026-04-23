#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 (4h) AND close > 4h EMA50 (uptrend) AND volume > 1.8x 20-period MA.
Short when price breaks below S1 (4h) AND close < 4h EMA50 (downtrend) AND volume > 1.8x 20-period MA.
Exit when price returns to Camarilla H3/L3 levels or opposite extreme is hit.
Uses 4h for signal direction, 1h only for entry timing precision. Session filter (08-20 UTC) reduces noise.
Target: 15-30 trades/year per symbol with structure-based edge.
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
    
    # Pre-compute session hours (08-20 UTC) to avoid per-bar datetime ops
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Camarilla calculation: based on previous bar's range
    range_4h = high_4h - low_4h
    r1 = close_4h_arr + 0.125 * range_4h   # R1
    s1 = close_4h_arr - 0.125 * range_4h   # S1
    h3 = close_4h_arr + 0.25 * range_4h    # H3
    l3 = close_4h_arr - 0.25 * range_4h    # L3
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 4h EMA50 = uptrend, close < 4h EMA50 = downtrend
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: 1h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r1_aligned[i]  # Break above R1
        breakout_down = close[i] < s1_aligned[i]  # Break below S1
        return_to_h3 = close[i] < h3_aligned[i]  # Return below H3 (exit long)
        return_to_l3 = close[i] > l3_aligned[i]  # Return above L3 (exit short)
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above R1 AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: return to H3/L3 or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = return_to_h3 or opposite_extreme
            elif position == -1:
                exit_signal = return_to_l3 or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0