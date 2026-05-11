#!/usr/bin/env python3
"""
12h_RSI_Trend_Filter
Hypothesis: Trade long when price breaks above 12h RSI(14) oversold (30) and price > 1d EMA(50); short when below 70 and price < 1d EMA(50). Uses 1d trend filter to avoid counter-trend trades. RSI provides mean-reversion entries within the trend, reducing false signals. Targets 15-25 trades/year to minimize fee drag while capturing major swings in bull and bear markets.
"""

name = "12h_RSI_Trend_Filter"
timeframe = "12h"
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
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h RSI14 ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Filter (optional but recommended) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    rsi_oversold = 30
    rsi_overbought = 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA and RSI)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI crosses above oversold (30) + price > 1d EMA50 + volume confirmation
            if (rsi[i] > rsi_oversold and rsi[i-1] <= rsi_oversold and 
                close[i] > ema50_1d_aligned[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: RSI crosses below overbought (70) + price < 1d EMA50 + volume confirmation
            elif (rsi[i] < rsi_overbought and rsi[i-1] >= rsi_overbought and 
                  close[i] < ema50_1d_aligned[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: RSI crosses back through opposite level (50 for mean reversion)
            if position == 1:
                if rsi[i] < 50 and rsi[i-1] >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if rsi[i] > 50 and rsi[i-1] <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals