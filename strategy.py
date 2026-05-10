#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Overbought_Oversold_v2
# Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) on 4h to determine trend direction,
# combined with RSI(14) overbought/oversold levels for mean-reversion entries, and 1d EMA(34) as
# higher timeframe trend filter. Designed for low trade frequency with regime adaptation.
# Works in bull markets via trend-following and in bear markets via mean-reversion at extremes.

name = "4h_KAMA_Direction_RSI_Overbought_Oversold_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > er_length else 0
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        if position == 0:
            # Long: Price above KAMA (uptrend) AND RSI oversold (mean reversion)
            if price_above_kama and rsi_oversold and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) AND RSI overbought (mean reversion)
            elif price_below_kama and rsi_overbought and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price below KAMA (trend change) OR RSI overbought (take profit)
            if price_below_kama or rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price above KAMA (trend change) OR RSI oversold (take profit)
            if price_above_kama or rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals