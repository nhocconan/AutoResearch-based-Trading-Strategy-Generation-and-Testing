#!/usr/bin/env python3
"""
12h_Trix_ZeroCross_Trend_Volume
Hypothesis: TRIX (triple exponential moving average) zero-cross signals combined with 1w trend filter and 12h volume confirmation. 
TRIX captures momentum changes; 1w trend ensures alignment with long-term direction; volume filter avoids false signals. 
Designed for 15-30 trades/year per symbol with effective performance in both bull and bear markets.
"""

name = "12h_Trix_ZeroCross_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend (using close)
    close_1w = df_1w['close'].values
    ema1_1w = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2_1w = pd.Series(ema1_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3_1w = pd.Series(ema2_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Trend: price above/below weekly triple EMA
    trend_1w = ema3_1w  # Represents the smoothed trend
    
    # Align weekly trend to 12h timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get 12h data for TRIX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate TRIX (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then % change
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX as percentage change
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    # Handle first value
    trix[0] = 0
    
    # Align TRIX to 12h timeframe (already in 12h, but using align for consistency)
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # Get 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly trend (12), TRIX (15*3=45), volume EMA (20)
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trend_1w_aligned[i]) or 
            np.isnan(trix_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: price vs weekly trend EMA
        bullish_trend = close[i] > trend_1w_aligned[i]
        bearish_trend = close[i] < trend_1w_aligned[i]
        
        if position == 0:
            # Long: bullish weekly trend AND TRIX crosses above zero with volume
            if bullish_trend and trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish weekly trend AND TRIX crosses below zero with volume
            elif bearish_trend and trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR weekly trend turns bearish
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR weekly trend turns bullish
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals