#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1w EMA50 trend filter.
# Long when Williams %R < -80 (oversold) AND close > 1w EMA50 (uptrend).
# Short when Williams %R > -20 (overbought) AND close < 1w EMA50 (downtrend).
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. Williams %R identifies reversal points in ranging markets.
# 1w EMA50 ensures trading only with higher timeframe trend to avoid whipsaws in chop.
# 6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero (when high == low)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        williams_r = williams_r_aligned[i]
        ema50 = ema50_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Williams %R crosses above -50 (momentum fading)
            if williams_r > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses below -50 (momentum fading)
            if williams_r < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > EMA50 (uptrend)
            if (williams_r < -80) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) AND price < EMA50 (downtrend)
            elif (williams_r > -20) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dWilliamsR_Extreme_1wEMA50_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0