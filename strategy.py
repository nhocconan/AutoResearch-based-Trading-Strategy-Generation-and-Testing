#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX + volume spike + 1d choppiness regime filter
# Long when TRIX crosses above zero AND volume > 1.8x 20-period average AND 1d CHOP > 61.8 (ranging market for mean reversion)
# Short when TRIX crosses below zero AND volume > 1.8x 20-period average AND 1d CHOP > 61.8 (ranging market for mean reversion)
# Exit when TRIX crosses back through zero OR 1d CHOP < 38.2 (trending market)
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-35 trades/year per symbol.
# TRIX captures momentum with less whipsaw than MACD, volume spike confirms institutional participation,
# 1d choppiness regime ensures we only trade in ranging conditions where mean reversion works best.
# Performs well in both bull and bear markets by fading extremes in ranging markets.

name = "12h_TRIX_VolumeSpike_1dChoppiness_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA of EMA of EMA of close, then ROC)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change(periods=1).values * 100  # ROC of triple EMA
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high)-min(low))))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness index
    chop_1d = 100 * np.log10(sum_atr_14 / (np.log10(14) * range_14))
    
    # Align TRIX to 12h timeframe (no extra delay needed for momentum indicator)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Align 1d choppiness to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: TRIX crosses above zero AND volume spike AND 1d choppy (ranging) market
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume_filter[i] and 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX crosses below zero AND volume spike AND 1d choppy (ranging) market
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume_filter[i] and 
                  chop_1d_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses back below zero OR 1d market becomes trending
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or \
               chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses back above zero OR 1d market becomes trending
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or \
               chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals