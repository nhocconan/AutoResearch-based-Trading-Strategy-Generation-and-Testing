#!/usr/bin/env python3
"""
4h_GoldenRatio_StopReversal_12hTrend
Hypothesis: Uses 12h EMA for trend filter, golden ratio (0.618) retracement levels from swing highs/lows for entries, and volume confirmation. Designed for low trade frequency (<20/year) with high win rate by entering only at key retracement levels in trending markets. Works in bull/bear by following 12h trend direction.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate swing points (simplified: local highs/lows over 5 periods)
    def find_swing_points(arr, window=5):
        highs = np.full_like(arr, np.nan)
        lows = np.full_like(arr, np.nan)
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                highs[i] = arr[i]
            if arr[i] == np.min(arr[i-window:i+window+1]):
                lows[i] = arr[i]
        return highs, lows
    
    # Find swing highs and lows on close prices
    swing_highs, swing_lows = find_swing_points(close, 5)
    
    # Forward fill swing points to get most recent levels
    def ffill_nan(arr):
        mask = np.isnan(arr)
        if not np.any(mask):
            return arr
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        return arr[idx]
    
    swing_highs_ff = ffill_nan(swing_highs)
    swing_lows_ff = ffill_nan(swing_lows)
    
    # Calculate golden ratio retracement levels (0.618)
    # For uptrend: retracement from swing low to swing high
    # For downtrend: retracement from swing high to swing low
    diff = swing_highs_ff - swing_lows_ff
    retracement_618 = swing_lows_ff + 0.618 * diff
    
    # Volume confirmation: volume > 1.5x average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(swing_highs_ff[i]) or
            np.isnan(swing_lows_ff[i]) or
            np.isnan(retracement_618[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_conf = vol_confirm[i]
        
        # Entry conditions: golden ratio retracement with trend and volume
        long_entry = (close[i] <= retracement_618[i]) and uptrend and vol_conf
        short_entry = (close[i] >= retracement_618[i]) and downtrend and vol_conf
        
        # Exit conditions: opposite retracement level or trend reversal
        long_exit = (close[i] >= swing_highs_ff[i]) or (not uptrend)
        short_exit = (close[i] <= swing_lows_ff[i]) or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_GoldenRatio_StopReversal_12hTrend"
timeframe = "4h"
leverage = 1.0