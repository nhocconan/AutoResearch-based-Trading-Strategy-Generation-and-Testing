#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_RegimeFilter
Hypothesis: TRIX momentum combined with volume spikes and Choppiness regime filter works in both bull and bear markets.
Uses 4h TRIX(12) crossing zero with volume > 2x 20-period average and Choppiness > 61.8 (ranging) for mean reversion or < 38.2 (trending) for trend following.
Targets 20-40 trades/year by requiring multiple confirmations to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX on close prices
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # Convert to percentage
    trix_values = trix.values
    
    # Zero line crossover signals
    trix_above_zero = trix_values > 0
    trix_below_zero = trix_values < 0
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    # Choppiness Index for regime detection
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high, 14) - min(low, 14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First TR is just high-low
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Regime thresholds
    chop_ranging = chop > 61.8   # Market is ranging (mean revert)
    chop_trending = chop < 38.2  # Market is trending (follow momentum)
    
    signals = np.zeros(n)
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_values[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_ranging[i]) or np.isnan(chop_trending[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions
        # In trending market: TRIX crosses above zero with volume spike
        # In ranging market: TRIX crosses above zero from negative with volume spike (mean reversion)
        long_signal = False
        if chop_trending[i]:
            # Trending market: follow TRIX momentum
            long_signal = (trix_values[i] > 0 and trix_values[i-1] <= 0 and volume_spike[i])
        elif chop_ranging[i]:
            # Ranging market: mean reversion from oversold
            long_signal = (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                          trix_values[i-1] < -0.5 and volume_spike[i])  # Oversold threshold
        
        # Short conditions
        # In trending market: TRIX crosses below zero with volume spike
        # In ranging market: TRIX crosses below zero from positive with volume spike (mean reversion)
        short_signal = False
        if chop_trending[i]:
            # Trending market: follow TRIX momentum
            short_signal = (trix_values[i] < 0 and trix_values[i-1] >= 0 and volume_spike[i])
        elif chop_ranging[i]:
            # Ranging market: mean reversion from overbought
            short_signal = (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                           trix_values[i-1] > 0.5 and volume_spike[i])  # Overbought threshold
        
        if long_signal:
            signals[i] = 0.25
        elif short_signal:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_TRIX_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0