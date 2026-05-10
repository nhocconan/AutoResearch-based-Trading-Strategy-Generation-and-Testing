#!/usr/bin/env python3
# 12h_PerformanceIndex_Momentum_1wTrend
# Hypothesis: Long when Performance Index crosses above 0 with volume > 1.3x average in uptrend (price > 1w EMA50).
# Short when Performance Index crosses below 0 with volume > 1.3x average in downtrend (price < 1w EMA50).
# Exit when PI crosses back across zero or ATR-based stoploss hit.
# Uses Performance Index (ROC-based momentum) to capture trend changes, works in both bull and bear markets.
# Designed for 12-37 trades/year to avoid fee drag.

name = "12h_PerformanceIndex_Momentum_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Performance Index: ROC(12) - ROC(25) normalized by ATR(14)
    roc12 = np.full(n, np.nan)
    roc25 = np.full(n, np.nan)
    for i in range(25, n):
        if close[i-12] != 0 and close[i-25] != 0:
            roc12[i] = (close[i] - close[i-12]) / close[i-12]
            roc25[i] = (close[i] - close[i-25]) / close[i-25]
    pi = roc12 - roc25
    # Normalize by ATR to make it scale-invariant
    pi_norm = np.full(n, np.nan)
    for i in range(25, n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            pi_norm[i] = pi[i] / atr[i]
    
    # Get 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(pi_norm[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1w EMA50 trend
            if close[i] > ema_50_1w_aligned[i]:  # Uptrend
                # Long: PI crosses above zero with volume confirmation
                if pi_norm[i] > 0 and pi_norm[i-1] <= 0 and volume[i] > 1.3 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: PI crosses below zero with volume confirmation
                if pi_norm[i] < 0 and pi_norm[i-1] >= 0 and volume[i] > 1.3 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: PI crosses back below zero or stoploss hit
            if pi_norm[i] < 0 or (i > 0 and low[i] < close[i-1] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: PI crosses back above zero or stoploss hit
            if pi_norm[i] > 0 or (i > 0 and high[i] > close[i-1] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals