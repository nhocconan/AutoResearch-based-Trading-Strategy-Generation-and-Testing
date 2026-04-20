#!/usr/bin/env python3
# 4h_1D_TRIX_VolumeSpike_Regime
# Hypothesis: On 4h timeframe, TRIX crossing above zero indicates bullish momentum, below zero bearish.
# Add volume spike (current volume > 2x 20-period average) and chop regime filter (CHOP(14) > 61.8 for mean reversion, < 38.2 for trend).
# Enter long when TRIX > 0 + volume spike + chop > 61.8 (oversold bounce in range).
# Enter short when TRIX < 0 + volume spike + chop < 38.2 (overbought reversal in trend).
# Exit on opposite TRIX cross.
# Uses 1d timeframe for chop calculation to avoid noise.
# Targets ~25-40 trades/year by requiring confluence of momentum, volume, and regime.
# Works in bull markets via trend-following and bear markets via mean reversion in ranges.

name = "4h_1D_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on 4h close (12-period EMA of EMA of EMA, then ROC)
    # EMA1
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix_raw = np.full_like(close, np.nan)
    trix_raw[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # Calculate 14-period chop from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    chop = np.full_like(close_1d, np.nan)
    mask = hh_ll > 0
    chop[mask] = 100 * np.log10(tr_sum[mask] / hh_ll[mask]) / np.log10(14)
    
    # Align TRIX and chop to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), trix_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Mean reversion in range (chop > 61.8): long on TRIX > 0 with volume spike
            if (chop_aligned[i] > 61.8 and 
                trix_aligned[i] > 0 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Trend reversal in trend (chop < 38.2): short on TRIX < 0 with volume spike
            elif (chop_aligned[i] < 38.2 and 
                  trix_aligned[i] < 0 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals