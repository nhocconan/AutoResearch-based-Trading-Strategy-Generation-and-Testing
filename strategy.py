#!/usr/bin/env python3
"""
Experiment #3111: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Weekly pivot levels (calculated from prior 1d candles) act as strong support/resistance. 
Breakouts above weekly R1 with price above weekly pivot and volume confirmation capture bullish momentum. 
Breakouts below weekly S1 with price below weekly pivot and volume confirmation capture bearish momentum. 
6h timeframe reduces noise while capturing multi-day swings. Position size 0.25. Target: 75-200 trades over 4 years.
Works in bull markets via continuation breakouts and in bear markets via mean reversion from extreme pivot levels (R4/S4).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3111_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot points (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points using prior week's 1d OHLC
    # Week = 7 trading days (approx, accounting for weekends)
    lookback_week = 7
    if len(close_1d) >= lookback_week:
        # Prior week's OHLC (exclude current forming week)
        prior_week_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().shift(1).values
        prior_week_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().shift(1).values
        prior_week_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).mean().shift(1).values
        
        # Weekly pivot calculation
        pp = (prior_week_high + prior_week_low + prior_week_close) / 3.0
        r1 = 2 * pp - prior_week_low
        s1 = 2 * pp - prior_week_high
        r2 = pp + (prior_week_high - prior_week_low)
        s2 = pp - (prior_week_high - prior_week_low)
        r3 = prior_week_high + 2 * (pp - prior_week_low)
        s3 = prior_week_low - 2 * (prior_week_high - pp)
        r4 = prior_week_high + 3 * (pp - prior_week_low)
        s4 = prior_week_low - 3 * (prior_week_high - pp)
        
        # Align to 6h timeframe
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pp_aligned = r1_aligned = s1_aligned = r4_aligned = s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = max(lookback, 20, 21)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long
                # Exit if price crosses below weekly pivot (mean reversion)
                if price < pp_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches weekly R4 (take profit at extreme)
                elif price >= r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price crosses above weekly pivot (mean reversion)
                if price > pp_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches weekly S4 (take profit at extreme)
                elif price <= s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND above weekly R1 AND above weekly pivot
            if (price > highest_high[i] and 
                price > r1_aligned[i] and 
                price > pp_aligned[i]):
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND below weekly S1 AND below weekly pivot
            elif (price < lowest_low[i] and 
                  price < s1_aligned[i] and 
                  price < pp_aligned[i]):
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals