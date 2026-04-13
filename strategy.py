#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 12h trend filter
    # Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
    # Long: Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA50 (uptrend)
    # Short: Bull Power < 0 AND Bear Power > 0 AND price < 12h EMA50 (downtrend)
    # Exit: Bull Power and Bear Power same sign (both >0 or both <0) indicating exhaustion
    # Uses 12h EMA for trend alignment to avoid counter-trend trades
    # Elder Ray measures bull/bear power relative to EMA; works in all regimes
    # Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fees
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 12h data for EMA trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 for 6h (Elder Ray base)
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_6h - ema_13_6h  # Bull Power = High - EMA
    bear_power = ema_13_6h - low_6h   # Bear Power = EMA - Low
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(13, n):  # start from 13 to have enough data for EMA13
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bull_positive = bull_power[i] > 0
        bear_positive = bear_power[i] > 0
        same_sign = (bull_positive and bear_positive) or (not bull_positive and not bear_positive)
        
        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions
        long_entry = bull_positive and (not bear_positive) and uptrend and position != 1
        short_entry = (not bull_positive) and bear_positive and downtrend and position != -1
        
        # Exit conditions (when bull and bear power same sign = exhaustion)
        exit_long = position == 1 and same_sign
        exit_short = position == -1 and same_sign
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
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

name = "6h_12h_elder_ray_trend_filter_v1"
timeframe = "6h"
leverage = 1.0