#!/usr/bin/env python3
"""
1d_1w_camarilla_breakout_volume_v1
Hypothesis: Use weekly EMA for trend bias and daily Camarilla levels for entries on 1d timeframe.
Long when daily close > daily R3 with volume confirmation and weekly bias up.
Short when daily close < daily S3 with volume confirmation and weekly bias down.
Exit when price reverses to opposite Camarilla level or weekly bias changes.
Designed to capture trend continuation and reversals at key levels with low trade frequency.
Target: 7-25 trades/year per symbol (28-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
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
    
    # Get daily data (same as primary timeframe for calculations)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]  # First value
    
    # Daily range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    r4 = close_1d + range_1d * 1.1 / 2
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align daily Camarilla levels to 1d timeframe (no alignment needed as same TF)
    r3_aligned = r3
    s3_aligned = s3
    r4_aligned = r4
    s4_aligned = s4
    
    # Weekly bias using EMA (13-period)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below daily S3 or weekly bias turns down
            if close[i] < s3_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above daily R3 or weekly bias turns up
            if close[i] > r3_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price closes above daily R3 with volume and weekly bias up
            if close[i] > r3_aligned[i] and vol_confirm[i] and close[i] > ema_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below daily S3 with volume and weekly bias down
            elif close[i] < s3_aligned[i] and vol_confirm[i] and close[i] < ema_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals