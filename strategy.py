#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + RSI mean reversion + 1d volume spike filter
# In trending markets: follow KAMA direction with RSI pullback entries
# In ranging markets: fade RSI extremes at support/resistance
# Uses 1d volume spike to confirm institutional participation
# Discrete sizing 0.25 targets 12-37 trades/year to minimize fee drag
# Works in bull/bear: trend following captures moves, mean reversion profits from chops

name = "12h_1d_kama_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h KAMA (ER=10, fastest=2, slowest=30)
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full(n, np.nan)
    kama[9] = close[9]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi])
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d volume spike (current > 2.0 * 20-period average)
    volume_spike_1d = volume_1d > 2.0 * avg_volume_1d
    
    # Align 1d indicators to 12h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if RSI > 70 (overbought) or price < KAMA (trend break)
            if rsi[i] > 70 or close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if RSI < 30 (oversold) or price > KAMA (trend break)
            if rsi[i] < 30 or close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on RSI < 30 (oversold) with volume spike and price > KAMA (uptrend)
            if rsi[i] < 30 and volume_spike_1d_aligned[i] and close[i] > kama[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on RSI > 70 (overbought) with volume spike and price < KAMA (downtrend)
            elif rsi[i] > 70 and volume_spike_1d_aligned[i] and close[i] < kama[i]:
                position = -1
                signals[i] = -0.25
    
    return signals