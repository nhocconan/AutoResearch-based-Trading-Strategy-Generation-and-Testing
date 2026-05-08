#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for TRIX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX calculation on 1d: triple smoothed EMA of ROC
    close_1d = df_1d['close'].values
    # ROC 1-period
    roc = np.diff(close_1d, prepend=close_1d[0]) / np.where(close_1d == 0, 1e-10, close_1d)
    # Triple EMA smoothing
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Align TRIX to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    
    # 1d EMA50 trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike detection: current volume > 2.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for TRIX calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_4h[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above zero with volume spike and 1d uptrend
            long_cond = (trix_4h[i] > 0 and trix_4h[i-1] <= 0 and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: TRIX crosses below zero with volume spike and 1d downtrend
            short_cond = (trix_4h[i] < 0 and trix_4h[i-1] >= 0 and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero (momentum fade)
            if trix_4h[i] < 0 and trix_4h[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero (momentum fade)
            if trix_4h[i] > 0 and trix_4h[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX momentum with volume confirmation and daily trend filter on 4h.
# TRIX (triple-smoothed ROC) filters noise and identifies sustained momentum.
# Entry on zero cross ensures trend confirmation. Volume spike >2.5x 20-period average
# confirms institutional participation. Trend filter aligns with daily bias.
# Works in bull markets (momentum continuation) and bear markets (mean reversion via exits).
# Target: 25-40 trades/year to minimize fee decay while capturing sustained moves.