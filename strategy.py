#!/usr/bin/env python3
# 12h_KAMA_Trend_With_RSI_and_Chop_Filter
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# In trending markets (ADX > 25), we follow KAMA direction with RSI(14) for momentum confirmation.
# In ranging markets (ADX < 20), we fade extremes using RSI < 30/ > 70.
# Uses 1d ADX for regime filter to avoid look-ahead. Target: 15-30 trades/year.

name = "12h_KAMA_Trend_With_RSI_and_Chop_Filter"
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        high_diff = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
        low_diff = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
    
    atr = np.zeros(len(df_1d))
    if len(tr) >= 14:
        atr[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = np.zeros(len(df_1d))
    minus_di = np.zeros(len(df_1d))
    dx = np.zeros(len(df_1d))
    if len(atr) >= 14 and atr[13] != 0:
        for i in range(14, len(df_1d)):
            if atr[i] != 0:
                plus_di[i] = 100 * (plus_dm[i] / atr[i])
                minus_di[i] = 100 * (minus_dm[i] / atr[i])
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros(len(df_1d))
    if len(dx) >= 14:
        adx[27] = np.mean(dx[14:28])
        for i in range(28, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h KAMA (10-period ER, 2/30 fast/slow)
    price_change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 10)))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])).reshape(-1, 10), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 12h indicators
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # Using 1d index for alignment (simplified)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d ADX (28), 12h KAMA/RSI (14)
    start_idx = max(28, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: trending (ADX > 25) or ranging (ADX < 20)
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Trending regime: follow KAMA direction
            if trending:
                if close[i] > kama_aligned[i] and rsi_aligned[i] > 50:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime: fade RSI extremes
            elif ranging:
                if rsi_aligned[i] < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi_aligned[i] > 70:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: trend breaks or RSI overbought in ranging
            if trending and (close[i] < kama_aligned[i] or rsi_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            elif ranging and rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or RSI oversold in ranging
            if trending and (close[i] > kama_aligned[i] or rsi_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            elif ranging and rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals