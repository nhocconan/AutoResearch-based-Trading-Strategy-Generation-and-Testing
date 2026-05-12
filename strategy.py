#!/usr/bin/env python3
name = "4h_KAMA_Trend_RSI_MeanReversion"
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
    
    # === 1d KAMA trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio (ER)
    change_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_1d = np.abs(np.diff(close_1d))
    er_1d = change_1d / (volatility_1d + 1e-10)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_1d = (er_1d * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === 1d RSI mean reversion filter ===
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + RSI oversold (<30)
            if (close[i] > kama_1d_aligned[i] and
                rsi_1d_aligned[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) + RSI overbought (>70)
            elif (close[i] < kama_1d_aligned[i] and
                  rsi_1d_aligned[i] > 70):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below KAMA or RSI overbought
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above KAMA or RSI oversold
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals