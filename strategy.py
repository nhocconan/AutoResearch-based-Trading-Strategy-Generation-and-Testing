#!/usr/bin/env python3
"""
4H_TRIX_VolumeSpike_ChopRegime
Hypothesis: 4h TRIX (triple EMA) momentum with volume spike confirmation and Choppiness Index regime filter.
Long when TRIX crosses above zero with volume spike in trending market (CHOP < 38.2).
Short when TRIX crosses below zero with volume spike in trending market (CHOP < 38.2).
Avoids ranging markets (CHOP > 61.8) to reduce whipsaw. Targets 20-50 trades/year on 4h timeframe.
"""
name = "4H_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate TRIX (triple EMA) on price
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then percent change
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # TRIX as percentage
    trix_values = trix.values
    
    # Calculate Choppiness Index (CHOP) on 1d timeframe for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of absolute price changes over 14 periods
    abs_close_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_close_change = pd.Series(abs_close_change).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR)/sum(|close change|)) / log10(14)
    chop = 100 * np.log10(atr * 14 / sum_abs_close_change) / np.log10(14)
    chop = np.where(sum_abs_close_change > 0, chop, 50)  # Avoid division by zero
    chop = np.where(atr > 0, chop, 50)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: current 4h volume > 2.0 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20)  # TRIX needs ~36 bars, volume avg needs 20
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trix_values[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending market filter: CHOP < 38.2 (strong trend)
            if chop_aligned[i] < 38.2:
                # Long: TRIX crosses above zero with volume spike
                if trix_values[i] > 0 and trix_values[i-1] <= 0 and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX crosses below zero with volume spike
                elif trix_values[i] < 0 and trix_values[i-1] >= 0 and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position != 0:
            # Exit: TRIX crosses zero in opposite direction (momentum reversal)
            if position == 1 and trix_values[i] < 0 and trix_values[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and trix_values[i] > 0 and trix_values[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals