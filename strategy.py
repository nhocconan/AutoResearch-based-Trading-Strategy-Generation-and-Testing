#!/usr/bin/env python3
# 12h_1w_1d_HTF_Confluence_Trend_Filter
# Hypothesis: Combines 1w trend (EMA50) with 1d momentum (RSI>50) and 12h volume confirmation.
# Uses 1w EMA50 for primary trend direction, 1d RSI for momentum filter, and 12h volume spike for entry confirmation.
# Designed for low turnover (target 15-25 trades/year) to minimize fee drag in 2025 ranging markets.
# Long when: price > 1w EMA50 AND 1d RSI > 50 AND volume > 1.5x 20-period EMA volume.
# Short when: price < 1w EMA50 AND 1d RSI < 50 AND volume > 1.5x 20-period EMA volume.
# Exit when trend reverses (price crosses 1w EMA50 in opposite direction).
# Uses 1w EMA50 as dynamic trend filter to avoid counter-trend trades in both bull and bear markets.

name = "12h_1w_1d_HTF_Confluence_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1w EMA50 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 1d RSI(14) Momentum Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_12h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of 1w EMA50 and 1d RSI warmup)
    start_idx = 100  # covers EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_12h[i]) or np.isnan(rsi_1d_12h[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Above 1w EMA50 + RSI > 50 + volume spike
            if (close[i] > ema50_1w_12h[i] and 
                rsi_1d_12h[i] > 50 and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Below 1w EMA50 + RSI < 50 + volume spike
            elif (close[i] < ema50_1w_12h[i] and 
                  rsi_1d_12h[i] < 50 and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Hold position until trend reverses
            if position == 1:
                # Exit: Price crosses below 1w EMA50
                if close[i] < ema50_1w_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above 1w EMA50
                if close[i] > ema50_1w_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals