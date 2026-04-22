#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1-day trend filter.
Long when Williams %R < -80 (oversold) and 1-day EMA(50) trending up.
Short when Williams %R > -20 (overbought) and 1-day EMA(50) trending down.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Williams %R identifies overextended moves; EMA filter ensures alignment with higher timeframe trend.
Works in both bull and bear markets by fading extremes in the direction of the 1-day trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 14:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend: 1 = up, -1 = down, 0 = unclear
    trend = np.zeros(len(ema_50_1d_aligned))
    trend[ema_50_1d_aligned > np.roll(ema_50_1d_aligned, 1)] = 1
    trend[ema_50_1d_aligned < np.roll(ema_50_1d_aligned, 1)] = -1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(willr[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold and 1-day trend up
            if willr[i] < -80 and trend[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Overbought and 1-day trend down
            elif willr[i] > -20 and trend[i] == -1:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if willr[i] > -50 and willr[i-1] <= -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50
                if willr[i] < -50 and willr[i-1] >= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0