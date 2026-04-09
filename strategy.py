#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend direction with 1d RSI filter and volume confirmation
# Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction
# Enters long when price > KAMA and RSI < 30 (oversold pullback in uptrend)
# Enters short when price < KAMA and RSI > 70 (overbought pullback in downtrend)
# Volume confirmation ensures institutional participation
# Discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: KAMA adapts to changing market conditions

name = "12h_1d_kama_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    # Wilder's smoothing for RSI
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Kaufman Adaptive Moving Average (KAMA)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, 10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    
    # Smoothing constants
    fastest = 2.0 / (2 + 1)   # EMA(2)
    slowest = 2.0 / (30 + 1)  # EMA(30)
    sc = np.square(er * (fastest - slowest) + slowest)
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start with first close
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align 1d indicators to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Volume confirmation: current volume > 1.5x average volume
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price crosses below KAMA or RSI becomes overbought
            if close[i] <= kama[i] or rsi_1d_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if price crosses above KAMA or RSI becomes oversold
            if close[i] >= kama[i] or rsi_1d_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above KAMA, RSI oversold (<30), volume confirmation
            if close[i] > kama[i] and rsi_1d_aligned[i] < 30 and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price below KAMA, RSI overbought (>70), volume confirmation
            elif close[i] < kama[i] and rsi_1d_aligned[i] > 70 and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals