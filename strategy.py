#!/usr/bin/env python3
"""
Hypothesis: 6-hour Bollinger Band breakout with 1-day Bollinger Band width regime filter.
Long when price breaks above upper BB(20,2) and 1-day BB width > 30th percentile (trending regime).
Short when price breaks below lower BB(20,2) and 1-day BB width > 30th percentile (trending regime).
Exit when price returns to middle band (mean reversion within trend).
Uses Bollinger Bands for volatility-based breakouts and regime filter to avoid chop.
Works in both bull and bear markets by only trading during trending regimes (high volatility).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Bollinger Bands for 6h period (20, 2)
    close_s = pd.Series(close)
    ma = close_s.rolling(window=20, min_periods=20).mean().values
    std = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = ma + 2 * std
    lower_bb = ma - 2 * std
    middle_bb = ma
    
    # Load 1-day data for BB width regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ma_1d = close_1d_s.rolling(window=20, min_periods=20).mean().values
    std_1d = close_1d_s.rolling(window=20, min_periods=20).std().values
    upper_bb_1d = ma_1d + 2 * std_1d
    lower_bb_1d = ma_1d - 2 * std_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / ma_1d  # Normalized width
    
    # 30th percentile of BB width over 50 periods for regime filter
    bb_width_percentile = pd.Series(bb_width_1d).rolling(window=50, min_periods=50).quantile(0.30).values
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or \
           np.isnan(bb_width_1d_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper BB and 1-day BB width > 30th percentile (trending)
            if close[i] > upper_bb[i] and bb_width_1d_aligned[i] > bb_width_percentile_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB and 1-day BB width > 30th percentile (trending)
            elif close[i] < lower_bb[i] and bb_width_1d_aligned[i] > bb_width_percentile_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Return to middle band (mean reversion within trend)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to or below middle band
                if close[i] <= middle_bb[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to or above middle band
                if close[i] >= middle_bb[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BB_Breakout_1dBBWidth_Regime"
timeframe = "6h"
leverage = 1.0