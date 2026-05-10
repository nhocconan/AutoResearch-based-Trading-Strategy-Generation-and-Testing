#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_ChoppyRegime_1wTrend
# Hypothesis: Uses TRIX momentum with volume spikes in choppy markets (CHOP>61.8) for mean reversion, filtered by weekly trend to avoid counter-trend trades. Designed for 4h to target 20-50 trades/year with high win rate. Works in bull/bear by aligning with weekly trend and regime filter.

name = "4h_TRIX_VolumeSpike_ChoppyRegime_1wTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate TRIX (12-period EMA of EMA of EMA of close, then ROC)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # first value undefined
    
    # Calculate Choppiness Index (CHOP) on daily timeframe
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    atr1[0] = 0
    atr2[0] = 0
    atr3[0] = 0
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop[0:13] = np.nan  # first 13 values undefined
    chop = np.nan_to_num(chop, nan=50.0)  # fill neutral
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get weekly EMA for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume average for confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 14, 20)  # Warmup for TRIX (36), chop (14), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation and chop regime filter (choppy = mean reversion)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        choppy_market = chop_aligned[i] > 61.8  # choppy/range bound
        
        if position == 0:
            # Long entry: TRIX turns up from oversold with volume in choppy market, weekly uptrend
            if trix[i] > trix[i-1] and trix[i-1] < -0.5 and volume_confirm and choppy_market and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX turns down from overbought with volume in choppy market, weekly downtrend
            elif trix[i] < trix[i-1] and trix[i-1] > 0.5 and volume_confirm and choppy_market and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns down or weekly trend turns down or market stops being choppy
            if trix[i] < trix[i-1] or not uptrend or chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns up or weekly trend turns up or market stops being choppy
            if trix[i] > trix[i-1] or not downtrend or chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals