#!/usr/bin/env python3
# 4h TRIX with Volume Spike and 12h Trend Filter
# Hypothesis: TRIX (triple-smoothed EMA) identifies momentum and trend changes.
# TRIX > 0 indicates bullish momentum, TRIX < 0 indicates bearish momentum.
# Combines with volume spike for momentum confirmation and 12h EMA50 trend filter.
# Designed for low trade frequency (~20-40/year) with clear entry/exit rules.
# Works in both bull and bear markets by following TRIX momentum.

name = "4h_TRIX_Volume_Trend"
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
    
    # === 12h Data for EMA Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === TRIX (15-period triple EMA) ===
    # Calculate TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = (ema3 - previous ema3) / previous ema3 * 100
    trix = np.full(n, np.nan)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX > 0 (bullish momentum) + volume spike + price above 12h EMA50
            if trix[i] > 0 and vol_spike[i] and close[i] > ema_50_4h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX < 0 (bearish momentum) + volume spike + price below 12h EMA50
            elif trix[i] < 0 and vol_spike[i] and close[i] < ema_50_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX turns negative (momentum shifts bearish)
            if trix[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive (momentum shifts bullish)
            if trix[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals