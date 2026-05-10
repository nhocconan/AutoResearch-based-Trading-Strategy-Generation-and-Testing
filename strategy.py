#!/usr/bin/env python3
# 4h_Keltner_Breakout_Trend_Volume
# Hypothesis: Keltner Channel breakouts with 1d EMA trend filter and volume confirmation provide consistent edge in both bull and bear markets.
# Uses Keltner Channel (ATR-based) for volatility-adaptive breakouts, EMA(50) on 1d for trend alignment, and volume > 1.5x average for confirmation.
# Designed for low trade frequency (target: 20-40/year) with discrete sizing (0.25) to minimize fee drag.

name = "4h_Keltner_Breakout_Trend_Volume"
timeframe = "4h"
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
    
    # Keltner Channel (20-period EMA, ATR multiplier 1.5)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 1.5 * atr
    kc_lower = ema_20 - 1.5 * atr
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_20[i]) or np.isnan(atr[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get 1d close for trend determination
        close_1d_series = pd.Series(close_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_series.values)
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Keltner Upper, volume confirmation, in uptrend
            if close[i] > kc_upper[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Keltner Lower, volume confirmation, in downtrend
            elif close[i] < kc_lower[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Keltner Middle (EMA20) or trend turns down
            if close[i] < ema_20[i] or is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Keltner Middle (EMA20) or trend turns up
            if close[i] > ema_20[i] or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals