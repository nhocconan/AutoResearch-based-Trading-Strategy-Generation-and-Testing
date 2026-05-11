#!/usr/bin/env python3
"""
1d_RSI_Overbought_Oversold_1wTrend
Hypothesis: Uses weekly trend filter with daily RSI extremes for mean reversion.
In long-term uptrend (price > weekly EMA200), enter long on RSI < 30 (oversold).
In long-term downtrend (price < weekly EMA200), enter short on RSI > 70 (overbought).
Exits when RSI crosses back to neutral (40-60 range). Designed for low trade frequency
(5-15 trades/year) to minimize fee drag while capturing mean reversion in trending markets.
Works in both bull and bear markets by aligning with weekly trend direction.
"""

name = "1d_RSI_Overbought_Oversold_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === DAILY RSI CALCULATION ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly EMA200 and RSI calculation)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: oversold in uptrend
            if rsi[i] < 30 and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought in downtrend
            elif rsi[i] > 70 and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or trend breaks
            if rsi[i] > 40 or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: RSI returns to neutral or trend breaks
            if rsi[i] < 60 or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals