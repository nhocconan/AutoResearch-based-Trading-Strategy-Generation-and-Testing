#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI20_80_VolumeFilter
# Hypothesis: KAMA identifies adaptive trend direction while avoiding whipsaw in choppy markets.
# RSI < 20 or > 80 captures extreme momentum exhaustion points for mean reversion entries.
# Volume confirmation filters low-liquidity false signals. Designed for 1d to reduce trade frequency
# and avoid fee drag, with robustness across bull/bear regimes via trend alignment and volatility filter.

name = "1d_KAMA_Trend_RSI20_80_VolumeFilter"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    # Align weekly KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly KAMA (uptrend) + RSI < 20 (oversold) + volume confirmation
            if close[i] > kama_aligned[i] and rsi[i] < 20 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly KAMA (downtrend) + RSI > 80 (overbought) + volume confirmation
            elif close[i] < kama_aligned[i] and rsi[i] > 80 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI > 70 (overbought) or price breaks below weekly KAMA
            if rsi[i] > 70 or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI < 30 (oversold) or price breaks above weekly KAMA
            if rsi[i] < 30 or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals