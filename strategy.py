#!/usr/bin/env python3
# 6h_Keltner_Breakout_Volume
# Hypothesis: Keltner Channel breakout with volume confirmation and 1d EMA50 trend filter.
# Uses 20-period ATR(10) for channel width. Long when price breaks above upper KC with
# volume > 1.5x 20-period average and price above 1d EMA50; short when breaks below lower
# KC with volume confirmation and price below 1d EMA50. Designed for 15-25 trades/year
# per symbol, works in bull via trend continuation and bear via mean reversion at extremes.

name = "6h_Keltner_Breakout_Volume"
timeframe = "6h"
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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(10) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # First bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Calculate EMA20 for KC middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: 2 * ATR above/below EMA20
    kc_upper = ema20 + 2.0 * atr
    kc_lower = ema20 - 2.0 * atr
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        ema20_val = ema20[i]
        kc_upper_val = kc_upper[i]
        kc_lower_val = kc_lower[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: price breaks above upper KC with volume confirmation and above 1d EMA50 trend
            if close[i] > kc_upper_val and close[i] > ema50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower KC with volume confirmation and below 1d EMA50 trend
            elif close[i] < kc_lower_val and close[i] < ema50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below EMA20 (middle of KC)
            if close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above EMA20 (middle of KC)
            if close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals