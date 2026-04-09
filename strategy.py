#!/usr/bin/env python3
# 4h_trix_volume_chop_v1
# Hypothesis: 4h strategy using TRIX (15-period) for momentum, volume confirmation (>1.5x 20-period average),
# and Choppiness Index regime filter (CHOP > 61.8 = ranging market for mean reversion).
# Long when TRIX crosses above zero in choppy market with volume confirmation.
# Short when TRIX crosses below zero in choppy market with volume confirmation.
# Uses 1d HTF data for Choppiness Index to avoid look-ahead, called ONCE before loop.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_volume_chop_v1"
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
    
    # 1d HTF data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for CHOP calculation
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Daily Choppiness Index (14-period)
    atr_d = np.zeros(len(high_d))
    for i in range(1, len(high_d)):
        tr = max(high_d[i] - low_d[i], abs(high_d[i] - close_d[i-1]), abs(low_d[i] - close_d[i-1]))
        atr_d[i] = (atr_d[i-1] * 13 + tr) / 14 if i > 1 else tr
    
    # Sum of ATR over 14 periods
    sum_atr_d = pd.Series(atr_d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    max_high_d = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    min_low_d = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(ATR) / (max_high - min_low)) / log10(14)
    chop_raw = 100 * np.log10(sum_atr_d / (max_high_d - min_low_d + 1e-10)) / np.log10(14)
    
    # Align daily Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # 4h TRIX (15-period): triple EMA of 1-period ROC
    roc = np.diff(close, prepend=close[0]) / close  # 1-period ROC
    ema1 = pd.Series(roc).ewm(span=15, min_periods=15, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, min_periods=15, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, min_periods=15, adjust=False).mean().values
    trix = ema3 * 100  # TRIX value
    
    # 4h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: CHOP > 61.8 = ranging market (mean reversion zone)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR chop regime ends
            if trix[i] < 0 or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR chop regime ends
            if trix[i] > 0 or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: TRIX crosses above zero
                if i > 0 and trix[i-1] <= 0 and trix[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: TRIX crosses below zero
                elif i > 0 and trix[i-1] >= 0 and trix[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals