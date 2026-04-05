#!/usr/bin/env python3
"""
exp_7555_6d_2025_06_05_v1
Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with 1-week EMA13 trend filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
In bull markets (price > weekly EMA13): go long when Bull Power turns positive with volume confirmation.
In bear markets (price < weekly EMA13): go short when Bear Power turns negative with volume confirmation.
Volume must be above 1.3x average to confirm strength.
Target: 50-150 trades over 4 years (12-37/year) with EMA13 smoothing reducing whipsaw.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7555_6d_2025_06_05_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_EMA = 13
WEEKLY_EMA_TREND = 13
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25

def generate_signals(prices):
    n = len(prices)
    if n < WEEKLY_EMA_TREND:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA13 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_13 = pd.Series(close_1w).ewm(span=WEEKLY_EMA_TREND, adjust=False, min_periods=WEEKLY_EMA_TREND).mean().values
    ema_1w_13_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_13)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA13 of close
    ema13 = pd.Series(close).ewm(span=ELDER_RAY_EMA, adjust=False, min_periods=ELDER_RAY_EMA).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(ELDER_RAY_EMA, WEEKLY_EMA_TREND, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_13_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Determine market regime from weekly trend
        bull_regime = close[i] > ema_1w_13_aligned[i]   # price above weekly EMA13
        bear_regime = close[i] < ema_1w_13_aligned[i]   # price below weekly EMA13
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Elder Ray signals with zero-cross detection
        bull_power_prev = bull_power[i-1] if i-1 >= 0 else 0
        bear_power_prev = bear_power[i-1] if i-1 >= 0 else 0
        
        # Bull power crosses above zero (bullish momentum building)
        bull_crossover = (bull_power[i] > 0) and (bull_power_prev <= 0)
        # Bear power crosses below zero (bearish momentum building)
        bear_crossover = (bear_power[i] < 0) and (bear_power_prev >= 0)
        
        # Entry conditions
        long_entry = bull_regime and bull_crossover and volume_confirmed
        short_entry = bear_regime and bear_crossover and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when bull power turns negative or regime changes
            if bull_power[i] <= 0 or not bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when bear power turns positive or regime changes
            if bear_power[i] >= 0 or not bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals