#!/usr/bin/env python3
"""
4h_Keltner_Breakout_1wTrend_Volume
Hypothesis: Enters long when 4h price breaks above upper Keltner Channel with upward 1w trend and volume spike.
Enters short when 4h price breaks below lower Keltner Channel with downward 1w trend and volume spike.
Uses Keltner Channel (ATR-based) for volatility-adjusted breakouts, 1w EMA for trend filter, and volume confirmation to avoid false breakouts.
Designed for low trade frequency (20-40 trades/year) to minimize fee flood and work in both bull and bear markets.
"""

name = "4h_Keltner_Breakout_1wTrend_Volume"
timeframe = "4h"
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
    
    # === 1W Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === 1D Data for ATR and Volume Average ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # True Range for ATR
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0]  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 20-period average volume on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 4h Keltner Channel (20, 2.0) ===
    tr_4h1 = np.abs(high - low)
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h1[0] = tr_4h2[0]
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_4h = pd.Series(tr_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    kc_upper = ma_20 + 2.0 * atr_4h
    kc_lower = ma_20 - 2.0 * atr_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # covers 20-period MA + ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper KC with uptrend and volume spike
            if (close[i] > kc_upper[i] and 
                close[i] > ema20_1w_aligned[i] and
                volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC with downtrend and volume spike
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema20_1w_aligned[i] and
                  volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below middle line (reversion to mean)
            if close[i] < ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above middle line
            if close[i] > ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals