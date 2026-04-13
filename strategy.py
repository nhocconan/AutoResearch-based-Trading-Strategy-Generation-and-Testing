#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    6h Williams %R Mean Reversion with Weekly Trend Filter
    Hypothesis: Williams %R identifies overbought/oversold conditions on 6h chart.
                Weekly trend filter ensures we trade with the higher timeframe momentum.
                Mean reversion works in both bull and bear markets as price oscillates around trend.
                Target: 60-120 trades over 4 years (15-30/year) to avoid fee drag.
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for weekly trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Williams %R(14) on 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_14 - close) / (highest_14 - lowest_14)
    
    # Align weekly EMA to 6h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_14[i]) or np.isnan(lowest_14[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Williams %R mean reversion signals
        # Long when oversold (< -80) in uptrend
        long_signal = (williams_r[i] < -80) and uptrend
        # Short when overbought (> -20) in downtrend
        short_signal = (williams_r[i] > -20) and downtrend
        
        # Exit when Williams %R returns to neutral zone (-50 center)
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1w_williams_r_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0