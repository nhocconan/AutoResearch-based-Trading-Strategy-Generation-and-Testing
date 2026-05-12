#!/usr/bin/env python3
# 4h_3BarReversal_1dTrend_Volume
# Hypothesis: 3-bar reversal pattern on 4h timeframe with 1d EMA trend filter and volume confirmation.
# Long: bullish 3-bar reversal (higher low + higher close) above 1d EMA50 with volume surge.
# Short: bearish 3-bar reversal (lower high + lower close) below 1d EMA50 with volume surge.
# Works in bull/bear by following higher timeframe trend while capturing short-term reversals.
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_3BarReversal_1dTrend_Volume"
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
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 3  # Need 3 bars for reversal pattern
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # 3-bar reversal patterns
        # Bullish: higher low + higher close over 3 bars
        bullish_reversal = (low[i] > low[i-2]) and (close[i] > close[i-2])
        # Bearish: lower high + lower close over 3 bars
        bearish_reversal = (high[i] < high[i-2]) and (close[i] < close[i-2])
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # LONG: bullish reversal, above 1d EMA50, volume confirmation
            if bullish_reversal and above_ema and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish reversal, below 1d EMA50, volume confirmation
            elif bearish_reversal and below_ema and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: bearish reversal or falls below 1d EMA50
            if bearish_reversal or not above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish reversal or rises above 1d EMA50
            if bullish_reversal or not below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals