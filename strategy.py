#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R with 1-day trend filter.
Long when %R < -80 (oversold) and 1-day close > 50-EMA (uptrend).
Short when %R > -20 (overbought) and 1-day close < 50-EMA (downtrend).
Exit when %R crosses -50 (mean reversion) or trend reverses.
Williams %R identifies reversal points in overbought/oversold conditions.
Trend filter ensures trading with the higher timeframe momentum.
Works in both bull and bear markets by capturing mean reversion within the prevailing trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R calculation (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Williams %R warmup
        # Skip if data not ready
        if np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold and uptrend on 1D
            if williams_r[i] < -80 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Overbought and downtrend on 1D
            elif williams_r[i] > -20 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: %R crosses above -50 (mean reversion) or trend turns down
                if williams_r[i] > -50 or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: %R crosses below -50 (mean reversion) or trend turns up
                if williams_r[i] < -50 or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0