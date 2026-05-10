#!/usr/bin/env python3
"""
1d_Keltner_Breakout_WeeklyTrend_Volume
Hypothesis: Keltner Channel breakout on daily timeframe in direction of weekly EMA34 trend with volume confirmation. Keltner adapts to volatility, making it robust in both bull and bear markets. Weekly trend filter ensures we trade with higher timeframe momentum, reducing whipsaws. Volume confirmation filters out low conviction breakouts. Target: 15-25 trades/year.
"""

name = "1d_Keltner_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_34_1w[i-1]
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily ATR(10) for Keltner Channel
    atr = np.full(n, np.nan)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    if len(tr) >= 10:
        atr[9] = np.nanmean(tr[:10])
        for i in range(10, n):
            atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]
    
    # Keltner Channel: EMA20 ± 2*ATR
    ema20 = np.full(n, np.nan)
    if len(close) >= 20:
        ema20[19] = np.mean(close[:20])
        alpha_ema = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha_ema * close[i] + (1 - alpha_ema) * ema20[i-1]
    
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Keltner + weekly EMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Close above Keltner upper band and above weekly EMA34
            if close[i] > kc_upper[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close below Keltner lower band and below weekly EMA34
            elif close[i] < kc_lower[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA20 (middle of Keltner)
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA20
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals