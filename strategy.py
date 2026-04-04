#!/usr/bin/env python3
"""
Experiment #2335: 6h Williams %R Extreme Reversal + Weekly Trend Filter
HYPOTHESIS: Williams %R identifies overextended moves on 6h timeframe. 
In strong weekly uptrend, extreme oversold (%R < -90) provides high-probability long entries.
In strong weekly downtrend, extreme overbought (%R > -10) provides high-probability short entries.
Weekly trend filter avoids counter-trend trades during major reversals. Works in both bull (buy dips) 
and bear (sell rallies) markets. Uses discrete sizing (0.25) to limit fee drag and ensure >50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2335_6h_williamsr_extreme_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    close = prices["close"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend direction
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_trend = np.where(close_1w > ema_1w, 1, -1)  # 1=uptrend, -1=downtrend
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # === 6h Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_14 - lowest_14
    williams_r = np.full(n, -50.0)  # default to neutral
    valid_range = hl_range != 0
    williams_r[valid_range] = ((highest_14[valid_range] - close[valid_range]) / hl_range[valid_range]) * -100
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = 50  # sufficient for weekly EMA and Williams %R
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_trend_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(highest_14[i]) or np.isnan(lowest_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: Williams %R returns to neutral zone (-50) or reverses extreme
            if position_side > 0:  # Long position
                if williams_r[i] > -50:  # Returned to neutral
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if williams_r[i] < -50:  # Returned to neutral
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        weekly_bias = weekly_trend_aligned[i]
        
        # Only trade in direction of weekly trend
        if weekly_bias > 0:  # Weekly uptrend - look for long entries
            # Extreme oversold: Williams %R < -90
            if williams_r[i] < -90:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif weekly_bias < 0:  # Weekly downtrend - look for short entries
            # Extreme overbought: Williams %R > -10
            if williams_r[i] > -10:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0  # No clear weekly trend
    
    return signals