#!/usr/bin/env python3
# 12h_1d_kama_rsi_volume_v1
# Strategy: KAMA trend + RSI momentum + volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in chop. RSI confirms momentum. Volume ensures participation.
# Works in bull by riding trends, in bear by avoiding false breaks and catching mean reversion at extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily RSI (14-period) for momentum filter
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA (10-period ER, 2/30 fast/slow) for trend
    er = np.abs(np.diff(close, prepend=0)) / (np.abs(np.diff(close, prepend=0)).rolling(window=10, min_periods=1).sum() + 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ok = volume > vol_avg  # Above average volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price > KAMA, RSI > 50, volume ok
        # Short: price < KAMA, RSI < 50, volume ok
        long_entry = close[i] > kama[i] and rsi_1d_aligned[i] > 50 and vol_ok[i]
        short_entry = close[i] < kama[i] and rsi_1d_aligned[i] < 50 and vol_ok[i]
        
        # Exit: opposite condition
        exit_long = position == 1 and (close[i] < kama[i] or rsi_1d_aligned[i] < 50)
        exit_short = position == -1 and (close[i] > kama[i] or rsi_1d_aligned[i] > 50)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals