#!/usr/bin/env python3
"""
Experiment #2935: 6h Williams %R Extreme + Weekly Trend Filter + Volume Confirmation
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe.
Only take trades when weekly trend (from 1w data) aligns: long when weekly close > weekly open (bullish),
short when weekly close < weekly open (bearish). Volume confirmation (>1.5x 20-period average) ensures
momentum behind the reversal. This captures mean reversion in strong trends while avoiding
counter-trend traps. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2935_6h_williamsr_extreme_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly trend bias (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly trend: bullish if weekly close > open, bearish if close < open
    weekly_bullish = close_1w > open_1w
    weekly_bullish_series = pd.Series(weekly_bullish)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish_series.values)
    
    # === 6h Indicators: Williams %R(14) ===
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full(n, np.nan)
    denominator = highest_high - lowest_low
    # Avoid division by zero
    valid = denominator != 0
    williams_r[valid] = ((highest_high[valid] - close[valid]) / denominator[valid]) * -100
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = max(lookback, 20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal or weekly trend change ---
        if in_position:
            # Exit conditions
            exit_long = False
            exit_short = False
            
            if position_side > 0:  # Long position
                # Exit if Williams %R becomes overbought OR weekly trend turns bearish
                if williams_r[i] > -20 or weekly_bullish_aligned[i] == False:
                    exit_long = True
            else:  # Short position
                # Exit if Williams %R becomes oversold OR weekly trend turns bullish
                if williams_r[i] < -80 or weekly_bullish_aligned[i] == True:
                    exit_short = True
            
            if exit_long or exit_short:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Get weekly trend bias
            is_bullish_weekly = weekly_bullish_aligned[i] == True
            
            # Long entry: Williams %R oversold (< -80) + bullish weekly trend
            if williams_r[i] < -80 and is_bullish_weekly:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short entry: Williams %R overbought (> -20) + bearish weekly trend
            elif williams_r[i] > -20 and not is_bullish_weekly:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals