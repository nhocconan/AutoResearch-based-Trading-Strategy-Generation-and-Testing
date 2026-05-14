#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_ATRStop_v1
Hypothesis: On 4h timeframe, trade long when price breaks above Camarilla R1 level and short when breaks below S1 level, 
filtered by 1d EMA34 trend and ATR-based stoploss. Camarilla levels provide intraday support/resistance derived from 
prior day's range. 1d EMA34 acts as higher-timeframe trend filter. ATR stoploss manages risk. 
Designed for 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.30) to minimize fee drag.
Works in bull/bear markets via 1d trend filter and volatility-based stops.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for prior 1d bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first bar
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - prev_close_1d), np.abs(low_1d - prev_close_1d)))
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values  # Wilder's ATR
    
    # Camarilla levels: based on prior day's range
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + 0.5 * hl_range_1d
    s1_1d = close_1d - 0.5 * hl_range_1d
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # ATR for stoploss calculation (4h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA34 (34), ATR (14), and Camarilla needs 1d data
    start_idx = max(34, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_34_val = ema_34_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1, above 1d EMA34
            long_signal = (close_val > r1_val) and (close_val > ema_34_val)
            
            # Short: price breaks below S1, below 1d EMA34
            short_signal = (close_val < s1_val) and (close_val < ema_34_val)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price breaks below S1 OR ATR stoploss (2*ATR below entry)
            if (close_val < s1_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price breaks above R1 OR ATR stoploss (2*ATR above entry)
            if (close_val > r1_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0