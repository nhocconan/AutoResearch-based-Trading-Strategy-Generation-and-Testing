#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX (15,9) with volume spike and choppiness regime filter
# TRIX = triple EMA momentum oscillator; zero-cross signals trend changes
# Long when TRIX crosses above zero AND TRIX rising AND volume > 1.5x 20-bar avg AND chop > 38.2 (trending regime)
# Short when TRIX crosses below zero AND TRIX falling AND volume > 1.5x 20-bar avg AND chop > 38.2
# Exit on opposite TRIX cross
# Uses discrete sizing 0.25 to limit fee churn. Target: 50-150 trades over 4 years.
# TRIX filters noise; volume confirms participation; chop filter ensures we trade only in trending markets.
# Works in bull (rising TRIX) and bear (falling TRIX) via symmetric long/short logic.

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(15,9): triple EMA of close, then ROC
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # TRIX = 100 * (EMA3 / EMA3.shift(1) - 1)
    trix_raw = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix_raw.values
    # Signal line: EMA of TRIX
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: >1.5x 20-bar average
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Choppiness Index: CHOP > 38.2 = trending regime (use 14-period)
    # CHOP = 100 * log10(sum(ATR(1),14) / (log10(highest high - lowest low,14)) * log10(14))
    atr_data = abs(high - low)
    atr_series = pd.Series(atr_data)
    atr_sum = atr_series.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    # Avoid division by zero
    chop = np.where(range_hl > 0, 100 * np.log10(atr_sum) / np.log10(range_hl) / np.log10(14), 50.0)
    chop_regime = chop > 38.2  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15*3, 20, 14, 9)  # TRIX warmup, volume, chop, signal
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        chop_trend = chop_regime[i]
        curr_trix = trix[i]
        curr_signal = trix_signal[i]
        prev_trix = trix[i-1]
        prev_signal = trix_signal[i-1]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TRIX crosses below signal line
            if curr_trix < curr_signal and prev_trix >= prev_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above signal line
            if curr_trix > curr_signal and prev_trix <= prev_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when TRIX crosses above signal line AND TRIX rising AND volume confirmation AND trending regime
            if curr_trix > curr_signal and prev_trix <= prev_signal and curr_trix > prev_trix and vol_conf and chop_trend:
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below signal line AND TRIX falling AND volume confirmation AND trending regime
            elif curr_trix < curr_signal and prev_trix >= prev_signal and curr_trix < prev_trix and vol_conf and chop_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals