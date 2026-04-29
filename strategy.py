#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX + volume spike + choppiness regime filter
# Long when TRIX crosses above zero AND choppy market (CHOP > 61.8) AND volume > 1.5x 20-bar avg
# Short when TRIX crosses below zero AND choppy market (CHOP > 61.8) AND volume > 1.5x 20-bar avg
# Uses 1d HTF for TRIX calculation and chop regime detection
# Discrete position sizing (0.25) to minimize fee churn
# Target: 15-30 trades/year on 12h timeframe (60-120 total over 4 years) to avoid overtrading
# Works in bull markets by capturing momentum and in bear markets by fading overextended moves
# during ranging conditions (chop > 61.8 indicates ranging market ideal for mean reversion)

name = "12h_TRIX_Chop_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX (1-period ROC of triple EMA) on 1d
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = pd.Series(ema3).pct_change(periods=1) * 100  # 1-period ROC
    trix = trix_raw.values
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate Choppiness Index on 1d
    atr_1d = pd.Series(np.maximum(np.maximum(high_1d - low_1d, 
                                              np.abs(high_1d - np.roll(close_1d, 1))),
                                  np.abs(low_1d - np.roll(close_1d, 1)))).rolling(
        window=14, min_periods=14).mean()
    sum_atr_14 = atr_1d.rolling(window=14, min_periods=14).sum()
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop_raw = 100 * np.log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = chop_raw.values
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 15, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_trix = trix_aligned[i]
        prev_trix = trix_aligned[i-1]
        curr_chop = chop_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero
            if curr_trix < 0 and prev_trix >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero
            if curr_trix > 0 and prev_trix <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only trade in choppy/ranging markets (CHOP > 61.8)
            if curr_chop > 61.8 and vol_conf:
                # Long when TRIX crosses above zero
                if curr_trix > 0 and prev_trix <= 0:
                    signals[i] = 0.25
                    position = 1
                # Short when TRIX crosses below zero
                elif curr_trix < 0 and prev_trix >= 0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals