#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 (1w) AND close > 1w EMA50 (uptrend) AND volume > 2.0x 20-period MA.
Short when price breaks below S1 (1w) AND close < 1w EMA50 (downtrend) AND volume > 2.0x 20-period MA.
Exit when price returns to Camarilla H4/L4 levels or opposite extreme is hit.
Designed for ~10-20 trades/year with structure-based edge in trending markets.
Uses 1d primary timeframe and 1w HTF for trend alignment to reduce overtrading and fee drag.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Camarilla calculation: based on previous day's range
    range_1d = high_1d - low_1d
    r1 = close_1d_arr + 1.0 * range_1d  # R1 = Close + 1.0 * (High - Low)
    s1 = close_1d_arr - 1.0 * range_1d  # S1 = Close - 1.0 * (High - Low)
    h4 = close_1d_arr + 1.125 * range_1d  # H4 = Close + 1.125 * (High - Low)
    l4 = close_1d_arr - 1.125 * range_1d  # L4 = Close - 1.125 * (High - Low)
    
    # Align Camarilla levels to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1w EMA50 = uptrend, close < 1w EMA50 = downtrend
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: 1d volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r1_aligned[i]  # Break above R1
        breakout_down = close[i] < s1_aligned[i]  # Break below S1
        return_to_h4 = close[i] < h4_aligned[i]  # Return below H4 (exit long)
        return_to_l4 = close[i] > l4_aligned[i]  # Return above L4 (exit short)
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above R1 AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0