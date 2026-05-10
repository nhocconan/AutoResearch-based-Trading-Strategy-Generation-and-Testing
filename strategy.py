#!/usr/bin/env python3
"""
1d_KeltnerBreakout_WeeklyTrend_Volume
Hypothesis: Keltner Channel breakout (ATR-based) with weekly trend filter (EMA34) and volume confirmation.
Breakouts occur at channel boundaries; weekly EMA34 filters for trend direction. Volume confirms strength.
Works in bull (breakouts above upper channel with up trend) and bear (breakouts below lower channel with down trend).
Target: 20-60 total trades over 4 years (5-15/year).
"""

name = "1d_KeltnerBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # ATR(10) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    if n >= 10:
        atr[9] = np.nanmean(tr[1:11])
        for i in range(10, n):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # EMA20 for Keltner Channel middle line
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        alpha20 = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha20 * close[i] + (1 - alpha20) * ema20[i-1]
    
    # Keltner Channel: Upper = EMA20 + 2*ATR, Lower = EMA20 - 2*ATR
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Volume SMA20
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 10)  # warmup for weekly EMA34, EMA20, ATR
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        if position == 0:
            # Long: Close breaks above upper Keltner Channel with weekly uptrend and volume
            if close[i] > kc_upper[i] and close[i] > ema34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Keltner Channel with weekly downtrend and volume
            elif close[i] < kc_lower[i] and close[i] < ema34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close crosses below EMA20 (middle of channel) or weekly trend turns down
            if close[i] < ema20[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close crosses above EMA20 or weekly trend turns up
            if close[i] > ema20[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals