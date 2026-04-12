#!/usr/bin/env python3
"""
4h_1d_Liquidity_Zone_Breakout_v1
Hypothesis: Trade breakouts from 1d liquidity zones (equal highs/lows) with volume confirmation and 4h trend filter.
Works in bull/bear via trend filter. Targets 20-40 trades/year by requiring multiple confluence factors.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Liquidity_Zone_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def find_equal_levels(arr, tolerance_pct=0.003):
    """Find equal highs/lows within tolerance percentage"""
    n = len(arr)
    if n < 2:
        return np.zeros(n, dtype=bool)
    
    equal_levels = np.zeros(n, dtype=bool)
    for i in range(1, n-1):
        # Check if current high/low equals previous within tolerance
        if abs(arr[i] - arr[i-1]) / arr[i-1] <= tolerance_pct:
            equal_levels[i] = True
            equal_levels[i-1] = True
    return equal_levels

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR LIQUIDITY ZONES ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Find daily equal highs/lows (liquidity zones)
    eq_highs = find_equal_levels(high_1d, 0.003)  # 0.3% tolerance
    eq_lows = find_equal_levels(low_1d, 0.003)
    
    # Build resistance/support levels from liquidity zones
    resistance = np.where(eq_highs, high_1d, np.nan)
    support = np.where(eq_lows, low_1d, np.nan)
    
    # Forward fill levels to create zones
    resistance_series = pd.Series(resistance)
    resistance_filled = resistance_series.ffill().bfill().values
    support_series = pd.Series(support)
    support_filled = support_series.ffill().bfill().values
    
    # === 4H TREND FILTER (EMA 34) ===
    if len(close) >= 34:
        ema_34 = np.zeros_like(close)
        ema_34[0] = close[0]
        alpha = 2.0 / (34 + 1)
        for i in range(1, len(close)):
            ema_34[i] = alpha * close[i] + (1 - alpha) * ema_34[i-1]
    else:
        ema_34 = np.full_like(close, np.nan)
    
    # Volume average (20-period)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(resistance_filled[i]) or np.isnan(support_filled[i]) or 
            np.isnan(ema_34[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter
        price_above_ema = close[i] > ema_34[i]
        price_below_ema = close[i] < ema_34[i]
        
        # Breakout entries at liquidity zones with volume and trend
        long_setup = (close[i] > resistance_filled[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < support_filled[i]) and vol_confirm and price_below_ema
        
        # Exit when price returns to opposite zone (mean reversion)
        exit_long = close[i] < support_filled[i]
        exit_short = close[i] > resistance_filled[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals