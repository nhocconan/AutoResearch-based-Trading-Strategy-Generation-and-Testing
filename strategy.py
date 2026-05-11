#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Regime_v1
Hypothesis: On daily timeframe, KAMA identifies trend direction, RSI identifies oversold/overbought conditions,
and Chop filters for trending vs ranging markets. In trending markets (Chop < 38.2), we follow KAMA direction.
In ranging markets (Chop > 61.8), we fade RSI extremes. This adapts to both bull and bear regimes.
Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe.
"""

name = "1d_KAMA_RSI_Chop_Regime_v1"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D Data for Indicators (already daily timeframe) ===
    # KAMA - Kaufman Adaptive Moving Average
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs correction
    # Recalculate properly
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop Index (14) - needs high/low
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14)
    atr = np.zeros_like(close)
    atr[13] = np.mean(tr[1:14])
    for i in range(14, len(close)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    atr_sum = np.zeros_like(close)
    for i in range(13, len(close)):
        if i == 13:
            atr_sum[i] = np.sum(atr[1:14])
        else:
            atr_sum[i] = atr_sum[i-1] - atr[i-13] + atr[i]
    
    # Max and Min range over 14 periods
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(len(close)):
        start_idx = max(0, i-13)
        max_high[i] = np.max(high[start_idx:i+1])
        min_low[i] = np.min(low[start_idx:i+1])
    
    # Chop calculation
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if atr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # === Weekly Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine market regime
            if chop[i] > 61.8:  # Ranging market
                # Fade RSI extremes
                if rsi[i] < 30:  # Oversold - go long
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:  # Overbought - go short
                    signals[i] = -0.25
                    position = -1
            elif chop[i] < 38.2:  # Trending market
                # Follow KAMA direction with weekly filter
                if close[i] > kama[i] and close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and close[i] < ema50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # In transition zone (38.2 <= Chop <= 61.8), stay aside
        elif position == 1:
            # Long exit conditions
            if chop[i] > 61.8 and rsi[i] > 70:  # Ranging + overbought
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and close[i] < kama[i]:  # Trending + below KAMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit conditions
            if chop[i] > 61.8 and rsi[i] < 30:  # Ranging + oversold
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and close[i] > kama[i]:  # Trending + above KAMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals