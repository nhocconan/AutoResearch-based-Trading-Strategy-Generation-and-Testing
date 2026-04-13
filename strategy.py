#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + Weekly Regime Filter
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    # Long: Bull Power > 0 AND Bear Power < 0 (strong bullish momentum) AND weekly trend up (price > weekly EMA34)
    # Short: Bear Power > 0 AND Bull Power < 0 (strong bearish momentum) AND weekly trend down (price < weekly EMA34)
    # Exit: Opposite power crosses zero (momentum exhaustion)
    # Weekly EMA34 provides structural bias that adapts to bull/bear markets
    # Elder Ray captures institutional buying/selling pressure
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA13 on 6h data for Elder Ray
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema13
    bear_power = ema13 - low_6h
    
    # Calculate weekly EMA34 for trend filter
    weekly_ema34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = close_1w > weekly_ema34
    weekly_trend_down = close_1w < weekly_ema34
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(34, n):  # start from 34 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Strong bullish momentum: Bull Power > 0 AND Bear Power < 0
        is_bullish_momentum = bull_power_val > 0 and bear_power_val < 0
        # Strong bearish momentum: Bear Power > 0 AND Bull Power < 0
        is_bearish_momentum = bear_power_val > 0 and bull_power_val < 0
        
        # Weekly trend filter
        is_weekly_up = weekly_trend_up_aligned[i] > 0.5
        is_weekly_down = weekly_trend_down_aligned[i] > 0.5
        
        # Entry conditions with weekly trend filter
        entry_long = is_bullish_momentum and is_weekly_up
        entry_short = is_bearish_momentum and is_weekly_down
        
        # Exit conditions: momentum exhaustion (opposite power crosses zero)
        exit_long = position == 1 and bear_power_val >= 0  # Bear power turns positive
        exit_short = position == -1 and bull_power_val >= 0  # Bull power turns positive
        
        # Execute signals
        if entry_long and position != 1:
            position = 1
            signals[i] = position_size
        elif entry_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_elder_ray_regime_filter_v1"
timeframe = "6h"
leverage = 1.0