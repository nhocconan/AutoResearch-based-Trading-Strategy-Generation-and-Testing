#!/usr/bin/env python3
"""
1d_1w_12h_atr_breakout_volume_v1
Hypothesis: Use weekly ATR-based breakout on daily timeframe with volume confirmation and weekly trend bias.
Works in bull markets by catching breakouts and in bear by avoiding false breakouts against trend.
Target: 8-20 trades/year per symbol (32-80 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_12h_atr_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and breakout calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 12h data for additional volume filter (optional)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ATR(14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA for trend bias
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Breakout levels: ±1.5 * ATR from open
    open_1d = df_1d['open'].values
    upper_break = open_1d + 1.5 * atr
    lower_break = open_1d - 1.5 * atr
    
    # Align breakout levels to daily timeframe (they're already daily, but ensure alignment)
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    
    # Volume confirmation: volume > 1.3x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below open or weekly trend turns down
            if close[i] < open_1d[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above open or weekly trend turns up
            if close[i] > open_1d[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with volume and weekly bias up
            if close[i] > upper_break_aligned[i] and vol_confirm[i] and close[i] > ema_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume and weekly bias down
            elif close[i] < lower_break_aligned[i] and vol_confirm[i] and close[i] < ema_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals