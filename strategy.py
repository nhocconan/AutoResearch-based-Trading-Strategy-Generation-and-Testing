#!/usr/bin/env python3
# 6h_KAMA_RSI_Trend_With_1dVolatility_Regime
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in choppy markets. Combined with RSI momentum and 1-day volatility regime (low ATR = range, high ATR = trend), this strategy captures trending moves while avoiding false signals in low-volatility environments. Works in both bull and bear markets by using volatility regime to filter entries and KAMA/RSI for momentum confirmation. Targets 50-150 total trades over 4 years.

timeframe = "6h"
name = "6h_KAMA_RSI_Trend_With_1dVolatility_Regime"
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
    
    # 1d data for volatility regime and context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=1))
    abs_change = np.abs(np.diff(close, n=1))
    # For ER calculation: |close[t] - close[t-10]| / sum(|close[i] - close[i-1]| for i=1..10)
    change_10 = np.zeros_like(close)
    abs_change_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        change_10[i] = np.abs(close[i] - close[i-10])
        # Sum of absolute changes over last 10 periods
        abs_change_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1], n=1)))
    
    er = np.zeros_like(close)
    er[10:] = change_10[10:] / np.where(abs_change_sum[10:] == 0, 1, abs_change_sum[10:])
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day ATR for volatility regime
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: high ATR = trending market, low ATR = ranging
    volatility_regime = atr_1d > atr_ma_1d  # True when volatility is above its 50-period average
    
    # Align 1d indicators to 6h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    volatility_regime_aligned = align_htf_to_ltf(prices, df_1d, volatility_regime)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for indicators to stabilize
        # Skip if any critical value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(volatility_regime_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, high volatility regime (trending), volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                volatility_regime_aligned[i] and
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, high volatility regime (trending), volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  volatility_regime_aligned[i] and
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or RSI < 40
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or RSI > 60
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals