#!/usr/bin/env python3
"""
12h_Keltner_Channel_Breakout_1dTrend_Volume
Hypothesis: Keltner Channel breakout on 12h combined with 1d EMA34 trend filter and volume confirmation.
Keltner Channels (EMA-based ATR bands) adapt to volatility, reducing false breakouts in ranging markets.
Trend filter ensures alignment with higher timeframe momentum. Volume confirmation filters weak breakouts.
Works in bull markets (upward breakouts with volume) and bear markets (downward breakouts with volume).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_Keltner_Channel_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Keltner Channel parameters (20-period EMA, 2x ATR)
    ema20 = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    if n >= 20:
        # EMA20
        ema20[19] = np.mean(close[:20])
        alpha20 = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha20 * close[i] + (1 - alpha20) * ema20[i-1]
        # ATR (True Range)
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])
        atr[19] = np.nanmean(tr[1:20]) if np.sum(~np.isnan(tr[1:20])) >= 20 else np.nan
        for i in range(20, n):
            if np.isnan(tr[i]):
                atr[i] = atr[i-1]
            else:
                atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and Keltner
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(ema20[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume (scaled from 1d)
        vol_12h_approx = vol_sma20_1d_aligned[i] / 2.0  # 1d volume / 2 for 12h approximation
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Close above upper Keltner band with uptrend and volume
            if close[i] > upper[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Keltner band with downtrend and volume
            elif close[i] < lower[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA20 (middle band) or trend reversal
            if close[i] < ema20[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA20 (middle band) or trend reversal
            if close[i] > ema20[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals