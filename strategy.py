#!/usr/bin/env python3
"""
Experiment #9691: 6h Donchian(20) Breakout + Daily ATR Regime + Volume Confirmation.
Hypothesis: Donchian breakouts with volume confirmation filtered by ATR-based regime 
(ATR ratio > 1.5 for breakouts, ATR ratio < 0.8 for mean reversion) provide robust 
performance across bull/bear markets. Targets 80-180 total trades over 4 years 
(20-45/year) to balance opportunity and cost. Uses 1d ATR for regime classification.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_9691_6h_donchian20_atr_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ATR_PERIOD = 14
ATR_RATIO_PERIOD = 30  # For ATR ratio calculation
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for ATR regime)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR for regime classification
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    
    # Calculate ATR ratio (current ATR / 30-period average ATR) for regime
    atr_ma_1d = pd.Series(atr_1d).rolling(window=ATR_RATIO_PERIOD, min_periods=ATR_RATIO_PERIOD).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    
    # Align 1d ATR ratio to 6h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    upper, lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management (6h)
    atr_6h = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ATR_RATIO_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(atr_ratio_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # ATR-based regime: high volatility (breakout) vs low volatility (mean reversion)
        high_volatility = atr_ratio_1d_aligned[i] > 1.5   # ATR > 1.5x average = breakout regime
        low_volatility = atr_ratio_1d_aligned[i] < 0.8    # ATR < 0.8x average = mean reversion regime
        
        # Breakout signals (high volatility): break above/below Donchian channels
        breakout_long = high_volatility and volume_spike and close[i] >= upper[i]
        breakout_short = high_volatility and volume_spike and close[i] <= lower[i]
        
        # Mean reversion signals (low volatility): fade at Donchian channels
        mean_rev_long = low_volatility and volume_spike and close[i] <= lower[i]
        mean_rev_short = low_volatility and volume_spike and close[i] >= upper[i]
        
        # Entry conditions
        long_entry = breakout_long or mean_rev_long
        short_entry = breakout_short or mean_rev_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_6h[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_6h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

# Note: align_htf_to_ltf is imported from mtf_data as align_htf_to_ltf
# Fixing the import/usage mismatch
from mtf_data import align_htf_to_ltf
# Replace align_htf_to_ltf calls (already correct in code above)