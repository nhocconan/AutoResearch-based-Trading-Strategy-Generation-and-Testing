#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_4hTrend_Filter
Hypothesis: On 1h timeframe, buy oversold conditions (RSI < 30) and sell overbought conditions (RSI > 70) only when aligned with 4h trend (EMA50). Uses RSI mean reversion in ranging markets and filters out counter-trend trades during strong trends. Designed for low trade frequency (<30/year) to minimize fee drag. Works in bull/bear by following 4h trend filter.
"""

name = "1h_RSI_MeanReversion_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h Data for Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1h RSI for Mean Reversion Signals ===
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers RSI and EMA50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) and price above 4h EMA50 (uptrend filter)
            if rsi[i] < 30 and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) and price below 4h EMA50 (downtrend filter)
            elif rsi[i] > 70 and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or stops below 4h EMA50
            if rsi[i] > 50 or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or stops above 4h EMA50
            if rsi[i] < 50 or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals