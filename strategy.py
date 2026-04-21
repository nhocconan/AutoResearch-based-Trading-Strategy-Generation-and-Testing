#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_Filter_v1
Hypothesis: On 1d timeframe, Donchian(20) breakouts aligned with weekly trend regime (price > weekly EMA34 for longs, < weekly EMA34 for shorts) capture sustained moves with reduced whipsaw. Weekly EMA34 acts as a strong trend filter, avoiding counter-trend breakouts. Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d Donchian(20) breakout levels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: highest high/lowest low of past 20 periods (including current?)
    # Using past 20 completed periods: lookback 20, so min_periods=20
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_hold_bars = 50  # max 50 days (~1.6 years) to avoid stale positions
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1w_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        
        # Weekly trend regime
        is_bull = price > weekly_ema
        is_bear = price < weekly_ema
        
        if position == 0:
            # Look for breakouts in direction of weekly trend
            if is_bull and price > upper_channel:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif is_bear and price < lower_channel:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions: opposite breakout or time stop
            if position == 1:
                if price < lower_channel:  # break below lower channel
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > upper_channel:  # break above upper channel
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            
            # Time-based exit
            if position != 0:
                # We don't track bars_since_entry simply; use price action for exit
                # Could add time stop but Donchian breakouts tend to trend
                pass
    
    return signals

name = "1d_Donchian20_WeeklyTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0