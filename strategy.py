#!/usr/bin/env python3
# 6h_12h_1d_consecutive_high_low_breakout_v1
# Hypothesis: Consecutive higher highs and lower lows on 12h and 1d timeframes define trend direction.
# On 6t, enter long when price breaks above recent high with volume confirmation during uptrend (higher highs/lows on 12h and 1d).
# Enter short when price breaks below recent low with volume confirmation during downtrend (lower highs/lows on 12h and 1d).
# Volume filter ensures institutional participation, reducing false breakouts.
# Works in both regimes by requiring alignment of higher timeframe structure.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_12h_1d_consecutive_high_low_breakout_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend structure
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate consecutive higher highs and higher lows (uptrend) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Higher high: current high > previous high
    hh_12h = np.zeros(len(df_12h), dtype=bool)
    hh_12h[1:] = high_12h[1:] > high_12h[:-1]
    
    # Higher low: current low > previous low
    hl_12h = np.zeros(len(df_12h), dtype=bool)
    hl_12h[1:] = low_12h[1:] > low_12h[:-1]
    
    # Uptrend: both higher high and higher low
    uptrend_12h = hh_12h & hl_12h
    
    # Calculate consecutive lower highs and lower lows (downtrend) on 12h
    # Lower high: current high < previous high
    lh_12h = np.zeros(len(df_12h), dtype=bool)
    lh_12h[1:] = high_12h[1:] < high_12h[:-1]
    
    # Lower low: current low < previous low
    ll_12h = np.zeros(len(df_12h), dtype=bool)
    ll_12h[1:] = low_12h[1:] < low_12h[:-1]
    
    # Downtrend: both lower high and lower low
    downtrend_12h = lh_12h & ll_12h
    
    # Calculate consecutive higher highs and higher lows (uptrend) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Higher high: current high > previous high
    hh_1d = np.zeros(len(df_1d), dtype=bool)
    hh_1d[1:] = high_1d[1:] > high_1d[:-1]
    
    # Higher low: current low > previous low
    hl_1d = np.zeros(len(df_1d), dtype=bool)
    hl_1d[1:] = low_1d[1:] > low_1d[:-1]
    
    # Uptrend: both higher high and higher low
    uptrend_1d = hh_1d & hl_1d
    
    # Calculate consecutive lower highs and lower lows (downtrend) on 1d
    # Lower high: current high < previous high
    lh_1d = np.zeros(len(df_1d), dtype=bool)
    lh_1d[1:] = high_1d[1:] < high_1d[:-1]
    
    # Lower low: current low < previous low
    ll_1d = np.zeros(len(df_1d), dtype=bool)
    ll_1d[1:] = low_1d[1:] < low_1d[:-1]
    
    # Downtrend: both lower high and lower low
    downtrend_1d = lh_1d & ll_1d
    
    # Align trend indicators to 6h timeframe (wait for 12h/1d bar to close)
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Recent high/low for breakout on 6h: 5-period lookback
    high_5 = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_5 = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    # Volume confirmation on 6h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.5
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(5, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if trend data is not available
        if (np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i])):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals during session with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 5-period low
            if close[i] < low_5[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 5-period high
            if close[i] > high_5[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 5-period high with volume
            # Require uptrend on both 12h and 1d
            if (close[i] > high_5[i] and 
                uptrend_12h_aligned[i] > 0.5 and 
                uptrend_1d_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 5-period low with volume
            # Require downtrend on both 12h and 1d
            elif (close[i] < low_5[i] and 
                  downtrend_12h_aligned[i] > 0.5 and 
                  downtrend_1d_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
    
    return signals