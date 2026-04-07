#!/usr/bin/env python3
"""
4h_atr_breakout_1w_trend_volume_v1
Hypothesis: On 4-hour timeframe, use weekly ATR breakout with weekly trend filter and volume confirmation to capture strong moves in both bull and bear markets. 
Enter long when price breaks above weekly high + 0.5*weekly ATR with volume > 1.5x average and price > weekly EMA50, short when price breaks below weekly low - 0.5*weekly ATR with volume > 1.5x average and price < weekly EMA50. 
Exit when price touches opposite weekly ATR level (weekly low - 0.5*ATR for long, weekly high + 0.5*ATR for short). 
Designed for low frequency (20-50 trades/year) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1w_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR, high/low, and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly ATR (14-period)
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # True Range
    tr1 = w_high - w_low
    tr2 = np.abs(w_high - np.roll(w_close, 1))
    tr3 = np.abs(w_low - np.roll(w_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    
    # Weekly high and low
    w_high_val = w_high
    w_low_val = w_low
    
    # Breakout levels: weekly high/low +/- 0.5*ATR
    breakout_high = w_high_val + 0.5 * atr_14
    breakout_low = w_low_val - 0.5 * atr_14
    
    # Weekly EMA50 for trend filter
    w_close_series = pd.Series(w_close)
    ema_50 = w_close_series.ewm(span=50, adjust=False).mean().values
    
    # Align to 4h timeframe
    breakout_high_aligned = align_htf_to_ltf(prices, df_1w, breakout_high)
    breakout_low_aligned = align_htf_to_ltf(prices, df_1w, breakout_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume average warmup
        # Skip if weekly data not available
        if np.isnan(breakout_high_aligned[i]) or np.isnan(breakout_low_aligned[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below breakout_low
            if close[i] <= breakout_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above breakout_high
            if close[i] >= breakout_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above breakout_high with volume confirmation AND price > weekly EMA50 (uptrend)
            long_entry = (close[i] > breakout_high_aligned[i]) and vol_confirm and (close[i] > ema_50_aligned[i])
            # Short entry: price breaks below breakout_low with volume confirmation AND price < weekly EMA50 (downtrend)
            short_entry = (close[i] < breakout_low_aligned[i]) and vol_confirm and (close[i] < ema_50_aligned[i])
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals