#!/usr/bin/env python3
# 4H_KAMA_1WTrend_AdaptiveSizing
# Hypothesis: 4h strategy using weekly KAMA direction with adaptive position sizing based on volatility regime.
# Uses weekly KAMA trend to filter direction, 4h RSI for momentum confirmation, and volatility-adjusted sizing.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend and reducing exposure in high volatility.
# Target: 4h timeframe with weekly HTF for trend filter.

name = "4H_KAMA_1WTrend_AdaptiveSizing"
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
    
    # Get weekly data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly KAMA (ER=10, fast=2, slow=30)
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)  # placeholder for true ER calculation
    # Simplified: use price change vs total movement for ER
    er = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        if i >= 10:
            price_change = np.abs(close_1w[i] - close_1w[i-10])
            total_change = np.sum(np.abs(np.diff(close_1w[i-10:i+1])))
            if total_change > 0:
                er[i] = price_change / total_change
            else:
                er[i] = 0
        else:
            er[i] = 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # 4h RSI for momentum confirmation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h ATR for volatility regime (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: normalize ATR by 50-period MA
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr / atr_ma  # >1 = high volatility, <1 = low volatility
    
    signals = np.zeros(n)
    
    start_idx = max(14, 50)  # RSI and volatility regime lookback
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_regime[i]) or atr_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Adaptive sizing: reduce exposure in high volatility
        base_size = 0.25
        vol_multiplier = np.clip(1.0 / vol_regime[i], 0.5, 1.5)  # inverse vol weighting
        size = base_size * vol_multiplier
        
        # Long: price above weekly KAMA (uptrend), RSI > 50 (bullish momentum)
        if close[i] > kama_aligned[i] and rsi[i] > 50:
            signals[i] = size
        # Short: price below weekly KAMA (downtrend), RSI < 50 (bearish momentum)
        elif close[i] < kama_aligned[i] and rsi[i] < 50:
            signals[i] = -size
        else:
            signals[i] = 0.0
    
    return signals