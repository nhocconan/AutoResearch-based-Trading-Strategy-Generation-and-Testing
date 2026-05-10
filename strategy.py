#!/usr/bin/env python3
# 4h_KAMA_Trend_Plus_RSI_Momentum
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a reliable trend signal in both bull and bear markets.
# Combined with RSI momentum and volume confirmation, this strategy aims to capture sustained moves while avoiding whipsaws in ranging markets.
# Uses 1d EMA for higher timeframe trend filter to ensure alignment with the dominant daily trend.
# Targets 20-40 trades per year to minimize fee drag and improve generalization.

name = "4h_KAMA_Trend_Plus_RSI_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA) ===
    # Fast EMA period = 2, Slow EMA period = 30
    change = np.abs(close - np.roll(close, 1))
    change[0] = 0  # First value has no prior
    
    # Direction over 10 periods
    direction = np.abs(close - np.roll(close, 10))
    direction[0:10] = 0  # Not enough data
    
    # Efficiency Ratio (ER)
    er = np.zeros_like(close)
    for i in range(10, n):
        if np.sum(change[i-9:i+1]) > 0:
            er[i] = direction[i] / np.sum(change[i-9:i+1])
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)      # EMA(2)
    slow_sc = 2 / (30 + 1)     # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # Same length as close
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # First average (simple mean)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder smoothing
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rs[14:] = avg_gain[14:] / np.where(avg_loss[14:] == 0, 1, avg_loss[14:])  # Avoid div by zero
    rsi = 100 - (100 / (1 + rs))
    
    # === HIGHER TIMEFRAME TREND FILTER (1d EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === VOLUME CONFIRMATION (20-period average) ===
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 34, 20)  # Warmup for KAMA, RSI, daily EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum: not overbought/oversold, but with momentum
        rsi_bullish = 50 < rsi[i] < 70
        rsi_bearish = 30 < rsi[i] < 50
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Higher timeframe trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI bullish momentum + volume + daily uptrend
            if price_above_kama and rsi_bullish and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI bearish momentum + volume + daily downtrend
            elif price_below_kama and rsi_bearish and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA OR RSI overbought OR daily trend turns down
            if close[i] <= kama[i] or rsi[i] >= 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA OR RSI oversold OR daily trend turns up
            if close[i] >= kama[i] or rsi[i] <= 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals