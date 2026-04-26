#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_ATRStop_v1
Hypothesis: On daily timeframe, trade breakouts above/below weekly Camarilla R3/S3 only when aligned with weekly EMA50 trend. Weekly Camarilla levels provide strong institutional support/resistance. Weekly EMA50 ensures trend alignment. Designed for 1d to capture major swing moves with very tight entries (target: 15-25 trades/year). Uses discrete sizing (0.25) to minimize fee drag and survive bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for Camarilla pivot and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use prior week's OHLC (shift by 1 to avoid look-ahead)
    high_prev = np.roll(high_1w, 1)
    low_prev = np.roll(low_1w, 1)
    close_prev = np.roll(close_1w, 1)
    # For first bar, use first available
    high_prev[0] = high_1w[0]
    low_prev[0] = low_1w[0]
    close_prev[0] = close_1w[0]
    
    # Camarilla calculations
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    r3 = close_prev + range_val * 1.1 / 4
    s3 = close_prev - range_val * 1.1 / 4
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align all HTF indicators to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for stoploss calculation (1d ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of pivot calc (1), EMA50 (50), ATR (14)
    start_idx = max(1, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
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
            # Exit: price breaks below S3 OR ATR stoploss (2.5*ATR below entry)
            if (close_val < s3_val) or (close_val < entry_price - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 OR ATR stoploss (2.5*ATR above entry)
            if (close_val > r3_val) or (close_val > entry_price + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0