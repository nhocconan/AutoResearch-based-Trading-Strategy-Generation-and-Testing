#!/usr/bin/env python3
"""
12h_1d_keltner_breakout_volume_v1
Hypothesis: 12-hour strategy using Keltner Channel breakouts with daily trend filter (EMA50) and volume confirmation.
Works in bull/bear by requiring breakouts to align with daily trend and confirming with volume to avoid false signals.
Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

name = "12h_1d_keltner_breakout_volume_v1"
timeframe = "12h"
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
    
    # Get daily data for trend and Keltner Channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous daily bar's range for Keltner Channel (ATR-based)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: EMA20 ± 2*ATR
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20_1d + 2 * atr
    lower_keltner = ema20_1d - 2 * atr
    
    # Align indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup for indicators
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price above daily EMA50 (uptrend) AND breaks above upper Keltner with volume
        if (close[i] > ema50_1d_aligned[i] and close[i] > upper_keltner_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below daily EMA50 (downtrend) AND breaks below lower Keltner with volume
        elif (close[i] < ema50_1d_aligned[i] and close[i] < lower_keltner_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to opposite Keltner band
        elif position == 1 and close[i] < lower_keltner_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > upper_keltner_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals