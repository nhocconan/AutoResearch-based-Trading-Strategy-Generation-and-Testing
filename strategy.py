#!/usr/bin/env python3
"""
Hypothesis: 4-hour KAMA trend filter with Bollinger Band mean-reversion entries and volume confirmation.
KAMA adapts to market noise, reducing whipsaw in choppy markets. Bollinger Bands provide dynamic
support/resistance for mean-reversion entries. Volume spikes confirm institutional interest.
This combination should work in both bull and bear regimes by adapting to market conditions.
Target: 20-35 trades/year per symbol.
"""

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
    
    # Load 4h data for KAMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on 4h close
    close_4h = pd.Series(df_4h['close'].values)
    # Efficiency Ratio
    change = abs(close_4h - close_4h.shift(10))
    volatility = abs(close_4h.diff()).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h.iloc[0]
    for i in range(1, len(close_4h)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_4h.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_4h = kama
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate Bollinger Bands (20, 2) on 4h close
    sma_20 = close_4h.rolling(window=20, min_periods=20).mean()
    std_20 = close_4h.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb.values)
    
    # Calculate 4h volume average (20-period)
    vol_4h = pd.Series(df_4h['volume'].values)
    vol_avg_20 = vol_4h.rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches lower BB with bullish KAMA trend and volume spike
            if (close[i] <= lower_bb_aligned[i] and 
                close[i] > kama_aligned[i] and 
                volume[i] > 2.0 * vol_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper BB with bearish KAMA trend and volume spike
            elif (close[i] >= upper_bb_aligned[i] and 
                  close[i] < kama_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to KAMA (dynamic mean)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above KAMA
                if close[i] >= kama_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below KAMA
                if close[i] <= kama_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_KAMA_BollingerBands_Volume"
timeframe = "4h"
leverage = 1.0