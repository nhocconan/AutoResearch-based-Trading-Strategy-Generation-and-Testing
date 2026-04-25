#!/usr/bin/env python3
"""
1d_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeConfirm
Hypothesis: On 1d timeframe, trade Camarilla H4/L4 breakouts in the direction of the 1w EMA50 trend with volume confirmation (>1.5x 20-bar average). Designed for low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drag. Works in bull markets via breakout continuation and in bear markets via shorting breakdowns. Uses 1w EMA50 for trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H4 = prev_close + 1.5 * prev_range
    L4 = prev_close - 1.5 * prev_range
    
    # Align 1d pivot levels to 1d timeframe (no shift needed as we use previous day's values)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1w EMA50 for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need enough for 1w EMA50 (50) and vol MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals in direction of 1w EMA50 trend
            # Long: price breaks above H4 AND above 1w EMA50
            long_breakout = (curr_close > H4_aligned[i]) and (curr_close > ema_50_1w_aligned[i])
            # Short: price breaks below L4 AND below 1w EMA50
            short_breakout = (curr_close < L4_aligned[i]) and (curr_close < ema_50_1w_aligned[i])
            
            long_entry = long_breakout and volume_spike[i]
            short_entry = short_breakout and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price reverts to mean (below H3) or breaks down (below L4)
            # Calculate H3 for exit
            H3_aligned = prev_close + 1.125 * prev_range
            H3_aligned = align_htf_to_ltf(prices, df_1d, H3_aligned)
            if curr_close < H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price reverts to mean (above L3) or breaks up (above H4)
            # Calculate L3 for exit
            L3_aligned = prev_close - 1.125 * prev_range
            L3_aligned = align_htf_to_ltf(prices, df_1d, L3_aligned)
            if curr_close > L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0