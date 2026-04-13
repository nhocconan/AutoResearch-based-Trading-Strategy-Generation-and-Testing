#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation
Hypothesis: Daily Camarilla pivot breakout with 4h volume confirmation. Long when price breaks above daily H3, short when breaks below L3. Volume filter avoids false breakouts. Designed for 1h timeframe to target 15-35 trades/year. Works in bull/bear via volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    range_ = prev_high - prev_low
    h3 = prev_close + (range_ * 1.1 / 6)
    l3 = prev_close - (range_ * 1.1 / 6)
    h4 = prev_close + (range_ * 1.1 / 2)
    l4 = prev_close - (range_ * 1.1 / 2)
    
    # Align to 1h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Get 4h volume for confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    vol_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Current 4h volume aligned
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position
    
    for i in range(30, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_4h_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume condition: current 4h volume > 1.5x 20-period average
        vol_condition = vol_4h_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        if position == 0:
            # Long breakout above H3
            if close[i] > h3_aligned[i] and vol_condition:
                position = 1
                signals[i] = position_size
            # Short breakout below L3
            elif close[i] < l3_aligned[i] and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price crosses L3 (reversal) or H4 (failed breakout)
            if close[i] < l3_aligned[i] or close[i] > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price crosses H3 (reversal) or L4 (failed breakout)
            if close[i] > h3_aligned[i] or close[i] < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0