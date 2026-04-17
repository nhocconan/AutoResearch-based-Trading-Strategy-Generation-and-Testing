#!/usr/bin/env python3
"""
Hypothesis: 1h EMA pullback strategy with 4h trend filter and 1d volatility regime.
Long when: price > 4h EMA50 (uptrend) AND price pulls back to 1h EMA21 AND 1d ATR percentile < 30 (low volatility).
Short when: price < 4h EMA50 (downtrend) AND price pulls back to 1h EMA21 AND 1d ATR percentile < 30 (low volatility).
Exit when price crosses 1h EMA8 in opposite direction.
Uses 4h for trend direction, 1h for entry timing and exit, 1d for volatility filter.
Designed to capture trend continuation pulls backs in low volatility environments. Target: 15-30 trades/year per symbol.
"""

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h EMAs for entry and exit
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d ATR(14) and its percentile rank (lookback 50 days)
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.roll(close_1d, 1)), 
                     np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    atr14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Calculate percentile rank of current ATR vs 50-day lookback
    atr_percentile = np.full_like(atr14, 50.0)  # default to median
    for i in range(50, len(atr14)):
        lookback = atr14[i-50:i]
        if len(lookback) > 0:
            atr_percentile[i] = (np.sum(lookback <= atr14[i]) / len(lookback)) * 100
    
    # Align all indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or
            np.isnan(ema8[i]) or
            np.isnan(ema21[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50 direction
        uptrend = close[i] > ema50_4h_aligned[i]
        downtrend = close[i] < ema50_4h_aligned[i]
        
        # Pullback condition: price near 1h EMA21 (within 0.5%)
        near_ema21 = abs(close[i] - ema21[i]) < 0.005 * close[i]
        
        # Volatility regime: low volatility (ATR percentile < 30)
        low_volatility = atr_percentile_aligned[i] < 30
        
        # Entry conditions
        if position == 0:
            # Long: uptrend + pullback to EMA21 + low volatility
            if uptrend and near_ema21 and low_volatility:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + pullback to EMA21 + low volatility
            elif downtrend and near_ema21 and low_volatility:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA8
            if close[i] < ema8[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above EMA8
            if close[i] > ema8[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA_Pullback_Trend_VolatilityFilter"
timeframe = "1h"
leverage = 1.0