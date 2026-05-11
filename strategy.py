#!/usr/bin/env python3
"""
6h_1d_WickReversal_Volume
Hypothesis: Price rejection at 1d high/low with long wicks signals reversal. 
Enter on 6h close beyond 1d Wick High/Low with volume confirmation. 
Works in bull/bear as mean-reversion at daily extremes. 
Target: 15-30 trades/year.
"""

name = "6h_1d_WickReversal_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for wick calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Wick Levels: Wick High = high - close, Wick Low = open - low ---
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    wick_high = high_1d - close_1d  # upper wick
    wick_low = open_1d - low_1d     # lower wick
    
    # Align to 6t
    wick_high_aligned = align_htf_to_ltf(prices, df_1d, wick_high)
    wick_low_aligned = align_htf_to_ltf(prices, df_1d, wick_low)
    
    # Wick High and Low levels in price
    wick_high_level = high_1d - wick_high_aligned  # = close_1d (but we keep for clarity)
    wick_low_level = low_1d + wick_low_aligned     # = open_1d
    
    # Actually, we want the actual price levels: Wick High = day high, Wick Low = day low
    # The wick size informs rejection strength
    day_high = high_1d
    day_low = low_1d
    day_high_aligned = align_htf_to_ltf(prices, df_1d, day_high)
    day_low_aligned = align_htf_to_ltf(prices, df_1d, day_low)
    
    # Wick strength ratio (for filtering strong rejections)
    body_size = np.abs(close_1d - open_1d)
    total_range = high_1d - low_1d
    wick_strength = (wick_high + wick_low) / (total_range + 1e-10)
    wick_strength_aligned = align_htf_to_ltf(prices, df_1d, wick_strength)
    
    # --- Volume Filter: above 1.5x median of last 24 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=24, min_periods=12).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = 24  # for volume median
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(day_high_aligned[i]) or np.isnan(day_low_aligned[i]) or 
            np.isnan(wick_strength_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Simple time-based exit: hold max 3 bars
                pass  # will be handled below
            continue
        
        if position == 0:
            # Look for rejection at daily high (strong upper wick) -> short
            # or rejection at daily low (strong lower wick) -> long
            if (high_6h[i] >= day_high_aligned[i] and 
                wick_strength_aligned[i] > 0.6 and  # strong wick rejection
                volume_6h[i] > vol_threshold[i]):
                # Price tested day high and rejected -> short
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
            elif (low_6h[i] <= day_low_aligned[i] and 
                  wick_strength_aligned[i] > 0.6 and  # strong wick rejection
                  volume_6h[i] > vol_threshold[i]):
                # Price tested day low and rejected -> long
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
        else:
            # Hold for max 3 bars or reverse signal
            # Track bars held
            if i < start_idx + 3:
                bars_held = 0
            else:
                # Simple: exit after 3 bars
                if position == 1:
                    signals[i] = 0.0
                    position = 0
                elif position == -1:
                    signals[i] = 0.0
                    position = 0
    
    return signals