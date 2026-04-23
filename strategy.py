#!/usr/bin/env python3
"""
Hypothesis: 1h mean reversion with 4h Camarilla pivot and 1d trend filter.
Long when price breaks below S1 (1d) AND close > 4h EMA20 (uptrend on 4h) AND volume > 1.5x 20-period MA.
Short when price breaks above R1 (1d) AND close < 4h EMA20 (downtrend on 4h) AND volume > 1.5x 20-period MA.
Exit when price returns to Camarilla H3/L3 levels.
Designed for ~20-30 trades/year by using 1h only for entry timing, 4h/1d for direction/structure.
Works in both bull/bear: mean reversion in ranges, trend filter avoids counter-trend in strong moves.
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
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's range
    range_1d = high_1d - low_1d
    r1 = close_1d_arr + 1.0 * (range_1d / 12) * 11  # R1 = C + 1.1*(H-L)
    s1 = close_1d_arr - 1.0 * (range_1d / 12) * 11  # S1 = C - 1.1*(H-L)
    h3 = close_1d_arr + 1.0 * (range_1d / 12) * 7   # H3 = C + 0.7*(H-L)
    l3 = close_1d_arr - 1.0 * (range_1d / 12) * 7   # L3 = C - 0.7*(H-L)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # need EMA20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 4h EMA20 = uptrend, close < 4h EMA20 = downtrend
        trend_up = close[i] > ema_20_4h_aligned[i]
        trend_down = close[i] < ema_20_4h_aligned[i]
        
        # Volume filter: 1h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_down = close[i] < s1_aligned[i]  # Break below S1
        breakout_up = close[i] > r1_aligned[i]    # Break above R1
        return_to_l3 = close[i] > l3_aligned[i]   # Return above L3 (exit short)
        return_to_h3 = close[i] < h3_aligned[i]   # Return below H3 (exit long)
        opposite_extreme = (position == 1 and breakout_up) or \
                           (position == -1 and breakout_down)
        
        if position == 0:
            # Long: Break below S1 AND uptrend AND volume confirmation
            if breakout_down and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break above R1 AND downtrend AND volume confirmation
            elif breakout_up and trend_down and vol_filter:
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

name = "1H_Camarilla_S1R1_MeanReversion_4hEMA20_Trend_VolumeFilter"
timeframe = "1h"
leverage = 1.0