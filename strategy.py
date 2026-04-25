#!/usr/bin/env python3
"""
4h_Camarilla_H4L4_Breakout_1dTrend_Volume
Hypothesis: On 4h timeframe, breakout above 1d Camarilla H4 or below L4 with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar avg). Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag. Works in bull/bear via trend filter. Uses discrete position sizing (0.30) to control drawdown.
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
    
    # 1d data for Camarilla pivots, EMA trend, and volume MA (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H4 = prev_close + 1.5 * prev_range
    L4 = prev_close - 1.5 * prev_range
    
    # Align 1d pivot levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current 4h volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and vol MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals
            long_breakout = (curr_close > H4_aligned[i]) and (curr_close > ema_34_aligned[i])
            short_breakout = (curr_close < L4_aligned[i]) and (curr_close < ema_34_aligned[i])
            
            long_entry = long_breakout and volume_spike[i]
            short_entry = short_breakout and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price crosses below EMA34 or re-enters Camarilla range
            if curr_close < ema_34_aligned[i] or curr_close < H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit when price crosses above EMA34 or re-enters Camarilla range
            if curr_close > ema_34_aligned[i] or curr_close > L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0