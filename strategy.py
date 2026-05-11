#!/usr/bin/env python3
name = "1d_Williams_Alligator_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # 1. Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 2. Williams Alligator on weekly: SMAs of median price
    median_price_1w = (df_1w['high'] + df_1w['low']) / 2
    jaw = median_price_1w.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = median_price_1w.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = median_price_1w.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align weekly Alligator lines to daily
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # 3. Weekly trend filter: bullish when lips > teeth > jaw
    weekly_bullish = lips_aligned > teeth_aligned
    weekly_bearish = lips_aligned < teeth_aligned
    
    # 4. Daily Williams Alligator for entry signals
    median_price = (high + low) / 2
    jaw_d = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_d = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_d = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 5. Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 6. Fixed position size to avoid churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(jaw_d[i]) or 
            np.isnan(teeth_d[i]) or np.isnan(lips_d[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        daily_bullish = lips_d[i] > teeth_d[i] and teeth_d[i] > jaw_d[i]
        daily_bearish = lips_d[i] < teeth_d[i] and teeth_d[i] < jaw_d[i]
        
        if position == 0:
            # Long: Daily bullish alignment + weekly bullish trend + volume spike
            if daily_bullish and weekly_bullish[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Daily bearish alignment + weekly bearish trend + volume spike
            elif daily_bearish and weekly_bearish[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - exit when Alligator lines cross or trend changes
            if position == 1:
                # Exit: Daily bearish crossover OR weekly trend turns bearish
                if lips_d[i] < teeth_d[i] or not weekly_bullish[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Daily bullish crossover OR weekly trend turns bullish
                if lips_d[i] > teeth_d[i] or not weekly_bearish[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals