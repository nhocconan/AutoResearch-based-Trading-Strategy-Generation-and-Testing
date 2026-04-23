#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX momentum + 1d volume spike + choppiness regime filter.
- TRIX(12) crossing zero line captures momentum shifts with reduced whipsaw vs MACD
- Volume > 2.0x 20-period average confirms conviction behind breakout
- Choppiness Index (CHOP) > 61.8 = ranging market (mean revert at Camarilla H3/L3)
- CHOP < 38.2 = trending market (follow TRIX crossovers)
- Long: TRIX crosses above zero + volume confirmation + CHOP < 38.2 (trending)
- Short: TRIX crosses below zero + volume confirmation + CHOP < 38.2 (trending)
- Exit: TRIX crosses zero in opposite direction OR CHOP > 61.8 and price touches Camarilla H3/L3
- Uses TRIX for clean momentum, volume for conviction, CHOP for regime adaptation
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy momentum in uptrend) and bear (sell momentum in downtrend)
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate TRIX(12) on close prices
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix_raw.values
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (atr_1d * 14)) / log10(14)
    # Avoid division by zero
    divisor = atr_1d * 14
    divisor = np.where(divisor == 0, 1e-10, divisor)
    chop_ratio = tr_sum_14 / divisor
    chop_ratio = np.where(chop_ratio <= 0, 1e-10, chop_ratio)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align TRIX, volume MA, and CHOP to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, prices, trix)  # Same timeframe, no alignment needed but for consistency
    vol_ma_aligned = align_htf_to_ltf(prices, prices, vol_ma)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d Camarilla levels for exit in ranging markets
    rng = high_1d - low_1d
    camarilla_h3 = close_1d + rng * (1.1 / 6)
    camarilla_l3 = close_1d - rng * (1.1 / 6)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(36, 20, 14)  # TRIX needs 36 (3*12), volume MA 20, CHOP 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_aligned[i-1]) or  # Need previous value for crossover
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma_aligned[i]
        
        # TRIX crossover signals
        trix_cross_above = trix_aligned[i-1] <= 0 and trix_aligned[i] > 0
        trix_cross_below = trix_aligned[i-1] >= 0 and trix_aligned[i] < 0
        
        # Regime filters
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: TRIX crosses above zero + volume confirmation + trending market
            if trix_cross_above and volume_confirm and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume confirmation + trending market
            elif trix_cross_below and volume_confirm and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR (ranging market and price touches H3)
            if trix_cross_below or (is_ranging and close[i] >= h3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR (ranging market and price touches L3)
            if trix_cross_above or (is_ranging and close[i] <= l3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_CHOP_Regime_CamarillaExit"
timeframe = "4h"
leverage = 1.0