#!/usr/bin/env python3
"""
6h_Advanced_Bollinger_Bands_1dTrend_Volume
Hypothesis: 6h Bollinger Bands (20,2) breakout in direction of 1d EMA34 trend, with volume confirmation and Bollinger width regime filter to avoid chop. Works in both bull and bear by following higher timeframe trend and using volatility-based entries. Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_Advanced_Bollinger_Bands_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get price, volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    # Bollinger Width for regime filter (avoid chop)
    bb_width = (bb_upper - bb_lower) / sma20
    # Use 50-period EMA of BB width to smooth
    bb_width_ema50 = pd.Series(bb_width).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), BBands (20), BB width EMA (50), volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i]) or
            np.isnan(bb_width_ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in low volatility regime (avoid chop)
        # BB width below its 50 EMA indicates low volatility/squeeze
        low_volatility = bb_width[i] < bb_width_ema50[i]
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above BB Upper with volume squeeze
            if close[i] > ema_34_aligned[i] and high[i] > bb_upper[i] and volume_filter[i] and low_volatility:
                signals[i] = 0.25
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below BB Lower with volume squeeze
            elif close[i] < ema_34_aligned[i] and low[i] < bb_lower[i] and volume_filter[i] and low_volatility:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below BB Middle OR trend turns bearish
            if low[i] < sma20[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above BB Middle OR trend turns bullish
            if high[i] > sma20[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals