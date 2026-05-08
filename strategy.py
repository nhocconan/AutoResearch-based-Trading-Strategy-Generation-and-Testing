#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Trend_with_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d KAMA for trend direction ===
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1d, 1))
    change = np.insert(change, 0, 0)  # align length
    volatility = np.abs(np.diff(close_1d, 1))
    volatility = np.insert(volatility, 0, 0)
    
    # Sum over 10 periods for ER
    change_sum = np.zeros_like(close_1d)
    volatility_sum = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        change_sum[i] = np.sum(change[i-9:i+1])
        volatility_sum[i] = np.sum(volatility[i-9:i+1])
    
    er = np.where(volatility_sum > 0, change_sum / volatility_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 12h RSI(14) for overbought/oversold ===
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI not overbought + volume confirmation
            if (close[i] > kama_12h_aligned[i] and 
                rsi[i] < 70 and 
                volume[i] > vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI not oversold + volume confirmation
            elif (close[i] < kama_12h_aligned[i] and 
                  rsi[i] > 30 and 
                  volume[i] > vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if close[i] < kama_12h_aligned[i] or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if close[i] > kama_12h_aligned[i] or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h KAMA trend following with RSI filter and volume confirmation.
# KAMA adapts to market noise - faster in trending markets, slower in ranging.
# Long when price > KAMA (uptrend), RSI < 70, volume above average.
# Short when price < KAMA (downtrend), RSI > 30, volume above average.
# Designed to work in both bull (follow KAMA trend) and bear (avoid false signals via RSI).
# Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag.
# Uses discrete sizing (0.25) to reduce churn. Works on BTC/ETH via adaptive trend.