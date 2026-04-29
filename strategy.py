#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(12) zero-line crossover with 1d volume spike confirmation and choppiness regime filter
# Long when TRIX crosses above zero AND 1d volume > 1.5x 20-bar avg AND 1d choppiness < 61.8 (trending)
# Short when TRIX crosses below zero AND 1d volume > 1.5x 20-bar avg AND 1d choppiness < 61.8 (trending)
# Exit when TRIX crosses back through zero (mean reversion of momentum failure)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# TRIX is a triple-smoothed EMA momentum oscillator that reduces noise and whipsaws.
# Volume confirmation ensures momentum has conviction, reducing false signals.
# Choppiness regime filter ensures we only trade in trending markets (avoids chop losses).
# Works in bull markets via upward momentum, works in bear via downward momentum with volume spikes.

name = "12h_TRIX12_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need sufficient data for TRIX and choppiness
        return np.zeros(n)
    
    # Calculate TRIX(12) on 12h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # Percentage change
    trix_values = trix.values
    
    # Calculate 1d volume spike confirmation: >1.5x 20-bar average volume
    volume_1d = df_1d['volume'].values
    volume_series_1d = pd.Series(volume_1d)
    volume_ma_20_1d = volume_series_1d.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 1.5 * volume_ma_20_1d
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for 1d
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate choppiness index
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    highest_lowest_diff = highest_high_14 - lowest_low_14
    chop_raw = 100 * np.log10(sum_atr_14 / (np.log10(14) * highest_lowest_diff))
    choppiness_1d = np.where(highest_lowest_diff > 0, chop_raw, 50)  # Avoid division by zero
    
    # Align 1d indicators to 12h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    choppiness_1d_aligned = align_htf_to_ltf(prices, df_1d, choppiness_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20, 14)  # Need sufficient history for TRIX (36), volume (20), chop (14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_values[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(choppiness_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike_1d_aligned[i] > 0.5  # Convert to boolean
        chop_filter = choppiness_1d_aligned[i] < 61.8  # Trending market
        trix_curr = trix_values[i]
        trix_prev = trix_values[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when TRIX crosses above zero AND volume spike AND trending market
            if trix_curr > 0 and trix_prev <= 0 and vol_spike and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below zero AND volume spike AND trending market
            elif trix_curr < 0 and trix_prev >= 0 and vol_spike and chop_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when TRIX crosses back below zero
            if trix_curr < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when TRIX crosses back above zero
            if trix_curr > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals