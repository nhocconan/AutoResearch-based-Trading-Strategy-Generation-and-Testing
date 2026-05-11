#!/usr/bin/env python3
"""
6h_HTF_Structure_Aligned_Trend
Hypothesis: Align with daily trend only when 6h price structure shows higher highs/lows (uptrend) or lower highs/lows (downtrend). Uses swing points from 6h swings filtered by 1d trend. Avoids chop by requiring clear structure. Designed for 15-35 trades/year per symbol to minimize fee drag while capturing real trends.
"""

name = "6h_HTF_Structure_Aligned_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # --- 1d EMA34 for trend filter ---
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 6h Swing Points (3-bar lookback/forward) ---
    # Swing High: higher high than 3 bars before and after
    # Swing Low: lower low than 3 bars before and after
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(3, n-3):
        if (high_6h[i] > high_6h[i-3] and high_6h[i] > high_6h[i-2] and 
            high_6h[i] > high_6h[i-1] and high_6h[i] > high_6h[i+1] and 
            high_6h[i] > high_6h[i+2] and high_6h[i] > high_6h[i+3]):
            swing_high[i] = True
        if (low_6h[i] < low_6h[i-3] and low_6h[i] < low_6h[i-2] and 
            low_6h[i] < low_6h[i-1] and low_6h[i] < low_6h[i+1] and 
            low_6h[i] < low_6h[i+2] and low_6h[i] < low_6h[i+3]):
            swing_low[i] = True
    
    # --- Trend Structure Detection ---
    # Uptrend: higher highs and higher lows
    # Downtrend: lower highs and lower lows
    # We'll track last swing high/low
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    
    for i in range(n):
        if swing_high[i]:
            last_high_val = high_6h[i]
        if swing_low[i]:
            last_low_val = low_6h[i]
        last_swing_high[i] = last_high_val
        last_swing_low[i] = last_low_val
    
    # --- Signal Generation ---
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after enough data for swings
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if EMA not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                # Simple stop: 2% adverse move
                if position == 1 and close_6h[i] <= entry_price * 0.98:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price * 1.02:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        is_uptrend = close_6h[i] > ema34_1d_aligned[i]
        is_downtrend = close_6h[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Look for structure-based entries
            if is_uptrend:
                # Long when we make a higher low and close above prior swing high
                if (not np.isnan(last_swing_low[i]) and not np.isnan(last_swing_high[i]) and
                    low_6h[i] > last_swing_low[i] and  # higher low
                    high_6h[i] > last_swing_high[i]):  # higher high
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_6h[i]
            elif is_downtrend:
                # Short when we make a lower high and close below prior swing low
                if (not np.isnan(last_swing_high[i]) and not np.isnan(last_swing_low[i]) and
                    high_6h[i] < last_swing_high[i] and  # lower high
                    low_6h[i] < last_swing_low[i]):     # lower low
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_6h[i]
        else:
            # Manage position: exit on structure break
            if position == 1:
                # Long: exit if we make a lower high (trend break)
                if (not np.isnan(last_swing_high[i]) and 
                    high_6h[i] < last_swing_high[i]):
                    signals[i] = 0.0
                    position = 0
                # Time-based exit: max 10 bars
                elif i >= 10 and entry_price > 0:  # rough check
                    # Simplified: exit after 5 periods if no clear signal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit if we make a higher low (trend break)
                if (not np.isnan(last_swing_low[i]) and 
                    low_6h[i] > last_swing_low[i]):
                    signals[i] = 0.0
                    position = 0
                # Time-based exit
                elif i >= 10 and entry_price > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals