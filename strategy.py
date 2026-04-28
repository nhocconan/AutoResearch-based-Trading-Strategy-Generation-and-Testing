#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(9) zero-cross + volume spike + 1d choppiness regime filter.
# TRIX(9) captures medium-term momentum with smoothing to reduce whipsaw.
# Volume spike (>2.0x 20-bar avg) confirms breakout strength.
# 1d Choppiness Index > 61.8 = ranging market (mean reversion), < 38.2 = trending.
# In ranging markets (CHOP > 61.8): fade TRIX extremes (long when TRIX crosses above -0.1, short when below +0.1).
# In trending markets (CHOP < 38.2): follow TRIX momentum (long when TRIX crosses above zero, short when below zero).
# Position size 0.25 balances return and drawdown.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in both bull and bear markets via regime adaptation.

name = "4h_Trix9_VolumeSpike_ChopperRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TRIX(9) on 4h close
    # TRIX = EMA(EMA(EMA(close, period), period), period) * 100
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # first value has no previous
    
    # Calculate 1d Choppiness Index (14)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # default to neutral
    
    # Align HTF chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d Choppiness Index
        is_ranging = chop_aligned[i] > 61.8   # ranging market (mean reversion)
        is_trending = chop_aligned[i] < 38.2   # trending market (trend follow)
        
        # TRIX zero-cross signals
        trix_cross_above_zero = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_below_zero = trix[i-1] >= 0 and trix[i] < 0
        trix_cross_above_minus01 = trix[i-1] <= -0.1 and trix[i] > -0.1
        trix_cross_below_plus01 = trix[i-1] >= 0.1 and trix[i] < 0.1
        
        # Entry logic based on regime
        long_entry = False
        short_entry = False
        
        if is_ranging:
            # In ranging market: mean reversion at TRIX extremes
            long_entry = trix_cross_above_minus01 and volume_spike[i]
            short_entry = trix_cross_below_plus01 and volume_spike[i]
        elif is_trending:
            # In trending market: follow TRIX momentum
            long_entry = trix_cross_above_zero and volume_spike[i]
            short_entry = trix_cross_below_zero and volume_spike[i]
        
        # Exit logic: opposite TRIX cross or volume dry-up
        long_exit = trix_cross_below_zero or not volume_spike[i]
        short_exit = trix_cross_above_zero or not volume_spike[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals