#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above R4 (1d) AND close > 1d EMA50 (uptrend) AND volume > 2.5x 20-period MA.
Short when price breaks below S4 (1d) AND close < 1d EMA50 (downtrend) AND volume > 2.5x 20-period MA.
Exit when price returns to Camarilla H3/L3 levels or opposite extreme is hit.
Uses tighter volume filter (2.5x vs 2.0x) and more extreme Camarilla levels (R4/S4) to reduce trades and improve edge.
Designed for ~15-25 trades/year with stronger structure-based edge in trending markets.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Camarilla calculation: based on previous day's range
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # H3 = Close + 0.75 * (High - Low)
    # L3 = Close - 0.75 * (High - Low)
    
    range_1d = high_1d - low_1d
    r4 = close_1d_arr + 1.5 * range_1d
    s4 = close_1d_arr - 1.5 * range_1d
    h3 = close_1d_arr + 0.75 * range_1d
    l3 = close_1d_arr - 0.75 * range_1d
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 4h volume > 2.5x 20-period MA (tighter filter)
        vol_filter = volume[i] > 2.5 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r4_aligned[i]  # Break above R4
        breakout_down = close[i] < s4_aligned[i]  # Break below S4
        return_to_h3 = close[i] < h3_aligned[i]  # Return below H3 (exit long)
        return_to_l3 = close[i] > l3_aligned[i]  # Return above L3 (exit short)
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above R4 AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below S4 AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R4S4_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0