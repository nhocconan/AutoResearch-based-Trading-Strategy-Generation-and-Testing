#!/usr/bin/env python3
# 1h_4h1d_Trend_Filter_Volume_Entry
# Hypothesis: Combine 4h trend (EMA21) and 1d momentum (ROC10) for direction, use 1h for entry timing with volume confirmation.
# Trend filter reduces false signals in chop, volume ensures participation. Designed for 15-30 trades/year to minimize fee drag.
# Works in bull/bear: trend filter adapts direction, volume avoids low-conviction moves.

name = "1h_4h1d_Trend_Filter_Volume_Entry"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Trend: EMA21 (direction filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d Momentum: ROC10 (momentum filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    roc_1d = np.full_like(close_1d, np.nan)
    roc_1d[10:] = (close_1d[10:] - close_1d[:-10]) / close_1d[:-10] * 100
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d)
    
    # === 1h Volume: 20-bar average (entry confirmation) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(roc_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]  # Volume spike
        
        if position == 0:
            # Long: 4h uptrend AND 1d positive momentum AND volume spike
            if ema_4h_aligned[i] > close[i] and roc_1d_aligned[i] > 0 and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend AND 1d negative momentum AND volume spike
            elif ema_4h_aligned[i] < close[i] and roc_1d_aligned[i] < 0 and vol_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend turns down OR volume dries up
            if ema_4h_aligned[i] < close[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h trend turns up OR volume dries up
            if ema_4h_aligned[i] > close[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals