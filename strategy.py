#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_ATRStop_v2
Hypothesis: On 12h timeframe, trade breakouts above/below daily Camarilla R1/S1 only when aligned with 1d EMA34 trend. Uses ATR-based stoploss and discrete sizing (0.25) to limit fee drag. Designed for 12h to capture swing moves with very tight entries (target: 12-37 trades/year). Uses Camarilla levels from daily timeframe for institutional support/resistance and 1d EMA34 for trend alignment. Avoids volume confirmation to reduce trade frequency and focus on price action + trend confluence.
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
    
    # Get 1d data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use prior day's OHLC (shift by 1 to avoid look-ahead)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    # For first bar, use first available
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    # Camarilla calculations
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    r1 = close_prev + range_val * 1.1 / 12
    s1 = close_prev - range_val * 1.1 / 12
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align all HTF indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for stoploss calculation (12h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of pivot calc (1), EMA34 (34), ATR (14)
    start_idx = max(1, 34, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1, above 1d EMA34
            long_signal = (close_val > r1_val) and (close_val > ema_34_val)
            
            # Short: price breaks below S1, below 1d EMA34
            short_signal = (close_val < s1_val) and (close_val < ema_34_val)
            
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
            # Exit: price breaks below S1 OR ATR stoploss (2.0*ATR below entry)
            if (close_val < s1_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR ATR stoploss (2.0*ATR above entry)
            if (close_val > r1_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_ATRStop_v2"
timeframe = "12h"
leverage = 1.0