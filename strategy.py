#!/usr/bin/env python3
# 6h_TRIX_VolumeSpike_ChopRegime_1dTrend
# Hypothesis: Use TRIX (15) on 6h for momentum, with 1d trend filter (EMA50), volume spike, and chop regime filter.
# TRIX crossing zero with volume confirms momentum shift. Chop filter avoids range-bound false signals.
# Works in bull/bear by aligning with 1d trend and using TRIX for precise momentum entries.
# Target: 20-40 trades/year.

name = "6h_TRIX_VolumeSpike_ChopRegime_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === TRIX (15) on 6h: triple EMA of ROC ===
    # ROC = (close - close.shift(1)) / close.shift(1)
    roc = np.diff(close, prepend=close[0]) / np.where(close[:-1] == 0, 1, close[:-1])
    roc = np.insert(roc, 0, 0)  # align length
    
    # Three successive EMAs
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3  # TRIX is the third EMA of ROC
    
    # === Chop regime filter (14-period) ===
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first TR
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)  # avoid div by zero
    chop = 100 * np.log10(sum_atr / range_hl) / np.log10(14)
    
    # === Volume spike (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(trix[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # TRIX momentum: crossing zero
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Chop regime: trending when < 38.2, range when > 61.8
        trending = chop[i] < 38.2
        
        if position == 0:
            # LONG: TRIX crosses up, volume, uptrend, trending regime
            if trix_cross_up and vol_ok and trend_up and trending:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses down, volume, downtrend, trending regime
            elif trix_cross_down and vol_ok and trend_down and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX crosses down or trend reversal
            if trix_cross_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses up or trend reversal
            if trix_cross_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals