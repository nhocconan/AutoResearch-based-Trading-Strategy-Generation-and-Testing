#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_ATRStop_v1
Hypothesis: On 1d timeframe, trade long when price breaks above Camarilla R3 level and short when breaks below S3 level, 
filtered by 1w EMA50 trend and ATR-based stoploss. Camarilla R3/S3 levels represent stronger support/resistance than R1/S1, 
reducing false breakouts. The 1w EMA50 trend filter ensures trades align with higher-timeframe direction, improving performance 
in both bull and bear markets. ATR stoploss manages risk. Targeting 30-100 total trades over 4 years (7-25/year) with 
discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True range for prior 1w bar
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = close_1w[0]  # first bar
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - prev_close_1w), np.abs(low_1w - prev_close_1w)))
    atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values  # Wilder's ATR
    
    # Camarilla levels: based on prior week's range (R3/S3 are stronger levels)
    hl_range_1w = high_1w - low_1w
    r3_1w = close_1w + 1.1666 * hl_range_1w  # R3 level
    s3_1w = close_1w - 1.1666 * hl_range_1w  # S3 level
    
    # Align HTF indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # ATR for stoploss calculation (1d ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), and Camarilla needs 1w data
    start_idx = max(50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_val = ema_50_1w_aligned[i]
        r3_val = r3_1w_aligned[i]
        s3_val = s3_1w_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R3, above 1w EMA50
            long_signal = (close_val > r3_val) and (close_val > ema_50_val)
            
            # Short: price breaks below S3, below 1w EMA50
            short_signal = (close_val < s3_val) and (close_val < ema_50_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S3 OR ATR stoploss (2*ATR below entry)
            if (close_val < s3_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 OR ATR stoploss (2*ATR above entry)
            if (close_val > r3_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0