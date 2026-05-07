#!/usr/bin/env python3
name = "4h_KAMA_Trend_With_RSI_Momentum"
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h KAMA for trend direction
    # Calculate KAMA: requires ER and smoothing constants
    change_12h = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility_12h = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) if False else np.nansum  # placeholder
    # Correct volatility calculation for KAMA
    volatility_12h = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        volatility_12h[i] = volatility_12h[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    # Reset volatility calculation properly
    volatility_12h = np.zeros(len(close_12h))
    for i in range(1, len(close_12h)):
        volatility_12h[i] = volatility_12h[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    
    # Efficiency Ratio (ER) for 10-period
    lookback_er = 10
    change_er = np.zeros(len(close_12h))
    volatility_er = np.zeros(len(close_12h))
    for i in range(lookback_er, len(close_12h)):
        change_er[i] = np.abs(close_12h[i] - close_12h[i-lookback_er])
        volatility_er[i] = volatility_12h[i] - volatility_12h[i-lookback_er]
    
    er = np.zeros(len(close_12h))
    for i in range(len(close_12h)):
        if volatility_er[i] > 0:
            er[i] = change_er[i] / volatility_er[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama_12h = np.zeros(len(close_12h))
    kama_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc[i] * (close_12h[i] - kama_12h[i-1])
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Load 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i < 14:
            avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i if i > 0 else gain[i]
            avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100  # avoid division by zero
    
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike filter on 4h: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for volume MA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend) AND RSI > 50 (momentum) AND volume spike
            if close[i] > kama_12h_aligned[i] and rsi_1d_aligned[i] > 50 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) AND RSI < 50 (momentum) AND volume spike
            elif close[i] < kama_12h_aligned[i] and rsi_1d_aligned[i] < 50 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below KAMA or RSI loses momentum
            if close[i] < kama_12h_aligned[i] or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA or RSI gains momentum
            if close[i] > kama_12h_aligned[i] or rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend filter combined with RSI momentum and volume confirmation.
# Long when price is above 12h KAMA (uptrend), RSI > 50 (bullish momentum), and volume spike.
# Short when price is below 12h KAMA (downtrend), RSI < 50 (bearish momentum), and volume spike.
# Uses discrete position size (0.25) to minimize churn. Target 20-40 trades/year.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI ensures we trade with momentum, not against it.
# Volume spike ensures conviction behind the move.
# Works in both bull and bear markets by following the trend with momentum confirmation.