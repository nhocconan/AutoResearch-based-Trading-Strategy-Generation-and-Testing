#!/usr/bin/env python3
"""
12h_KAMA_1dRSI_TrendFilter_V3
Strategy: 12h KAMA direction with 1d RSI filter and volume confirmation.
Long: KAMA bullish + RSI(14) < 30 + volume > 1.5x 20-period average
Short: KAMA bearish + RSI(14) > 70 + volume > 1.5x 20-period average
Exit: KAMA direction change
Position size: 0.25
Designed to catch mean reversion in extreme RSI zones during trending periods.
Timeframe: 12h
"""

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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |Change| / Volatility
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prev + SC * (price - prev)
    
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # For rolling volatility, we'll use sum of absolute changes over period
    vol_sum = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    er = np.where(vol_sum != 0, change / vol_sum, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 12h volume average (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(30, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: bullish if close > KAMA, bearish if close < KAMA
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # RSI extremes
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * vol_ma20[i])
        
        # Entry signals
        if position == 0:
            # Long: KAMA bullish + RSI oversold + volume
            if kama_bullish and rsi_oversold and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish + RSI overbought + volume
            elif kama_bearish and rsi_overbought and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns bearish
            if kama_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns bullish
            if kama_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_1dRSI_TrendFilter_V3"
timeframe = "12h"
leverage = 1.0