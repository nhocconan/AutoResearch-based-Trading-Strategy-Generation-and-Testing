#!/usr/bin/env python3
# 6h_Trix_Volume_Regime_Momentum
# Hypothesis: TRIX (12-period) crossing above/below zero with volume confirmation and regime filter (Choppiness Index < 61.8) captures momentum in trending markets while avoiding chop. Works in bull markets (TRIX > 0 with volume) and bear markets (TRIX < 0 with volume). Uses 1d Choppiness Index to filter regime. Target: 15-30 trades/year per symbol to minimize fee drag.

name = "6h_Trix_Volume_Regime_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Choppiness Index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # ATR (14-period smoothed)
    atr = np.zeros_like(tr)
    atr[13] = np.nanmean(tr[1:15])  # seed
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    atr_sum = np.zeros_like(atr)
    for i in range(13, len(atr)):
        if i == 13:
            atr_sum[i] = np.nansum(atr[1:15])
        else:
            atr_sum[i] = atr_sum[i-1] - atr[i-13] + atr[i]
    
    # High-Low range over 14 periods
    hh = np.zeros_like(high_1d)
    ll = np.zeros_like(low_1d)
    for i in range(len(high_1d)):
        if i < 13:
            hh[i] = np.nan
            ll[i] = np.nan
        else:
            hh[i] = np.nanmax(high_1d[i-13:i+1])
            ll[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]) or hh[i] == ll[i]:
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
    
    # Regime: chop < 61.8 = trending (favorable for momentum)
    chop_regime = chop < 61.8
    
    # Align 1d chop regime to 6h
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Calculate TRIX (12-period) on 6h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(chop_regime_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX > 0 (bullish momentum) + chop regime (trending) + volume spike
            if trix[i] > 0 and chop_regime_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX < 0 (bearish momentum) + chop regime (trending) + volume spike
            elif trix[i] < 0 and chop_regime_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX <= 0 or chop regime > 61.8 (choppy)
            if trix[i] <= 0 or not chop_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX >= 0 or chop regime > 61.8 (choppy)
            if trix[i] >= 0 or not chop_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals