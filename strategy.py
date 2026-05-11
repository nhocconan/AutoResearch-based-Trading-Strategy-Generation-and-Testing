#!/usr/bin/env python3
# 1d_70_30_RSI_1wTrend_Volume
# Hypothesis: 70/30 RSI on daily with 1-week EMA trend filter and volume confirmation.
# Works in bull markets (RSI>70 + uptrend) and bear markets (RSI<30 + downtrend).
# Low turnover (~10-20 trades/year) to minimize fee drag in ranging 2025 markets.

name = "1d_70_30_RSI_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 70/30 RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1-week EMA Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Volume Confirmation (20-day EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # covers RSI and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 70 + above weekly EMA50 + volume spike
            if (rsi[i] > 70 and 
                close[i] > ema50_1w_aligned[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: RSI < 30 + below weekly EMA50 + volume spike
            elif (rsi[i] < 30 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60)
            if position == 1:
                if rsi[i] < 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if rsi[i] > 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals