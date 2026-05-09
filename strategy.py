#!/usr/bin/env python3
# 6h_StellarCore_Trend_Momentum_Fusion
# Hypothesis: In both bull and bear markets, price tends to respect the 1d EMA(50) as dynamic support/resistance.
# Combines 60-period momentum (ROC) with 1d EMA trend filter to capture high-probability continuations.
# Uses volume surge (>1.5x 20-period average) to confirm institutional participation.
# Designed for low-frequency, high-conviction trades (~20-40/year) to minimize fee drag.
# Works in bull markets by buying pullbacks to EMA in uptrends, and in bear markets by selling bounces to EMA in downtrends.

name = "6h_StellarCore_Trend_Momentum_Fusion"
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
    
    # --- 1d EMA(50) for trend filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- 60-period ROC (momentum) ---
    lookback = 60
    roc = np.zeros(n)
    for i in range(lookback, n):
        if close[i - lookback] != 0:
            roc[i] = (close[i] - close[i - lookback]) / close[i - lookback] * 100
    
    # --- Volume surge filter: >1.5x 20-period average ---
    vol_ma_20 = np.zeros(n)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_surge = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            volume_surge[i] = volume[i] > 1.5 * vol_ma_20[i]
    
    # --- Signal generation ---
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # Ensure ROC and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(roc[i]) or 
            i < 20):  # volume_surge needs i>=20
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish momentum + price above 1d EMA50 + volume surge
            if roc[i] > 2.0 and close[i] > ema_50_1d_aligned[i] and volume_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + price below 1d EMA50 + volume surge
            elif roc[i] < -2.0 and close[i] < ema_50_1d_aligned[i] and volume_surge[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: momentum fades or price crosses below EMA
            if roc[i] < 0.5 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: momentum fades or price crosses above EMA
            if roc[i] > -0.5 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals