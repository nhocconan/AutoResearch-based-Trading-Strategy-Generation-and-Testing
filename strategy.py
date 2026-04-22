#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA + RSI + Choppiness filter (4h)
# Uses Kaufman's Adaptive Moving Average (KAMA) for trend direction,
# RSI(14) for momentum filtering, and Choppiness Index for regime detection.
# Long when KAMA rising + RSI > 50 + Chop < 61.8 (trending regime)
# Short when KAMA falling + RSI < 50 + Chop < 61.8
# Designed for 4h timeframe to target 20-50 trades/year per symbol.
# Works in bull/bear via KAMA trend + Chop regime filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast = 2
    slow = 30
    kama_period = 10
    
    # Direction
    change = np.abs(close - np.roll(close, kama_period))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    volatility = pd.Series(volatility).rolling(window=kama_period, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr, axis=1) / (max_high - min_low)) / np.log10(14)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: rising if current > previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + Chop < 61.8 + volume spike
            if (kama_rising and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + Chop < 61.8 + volume spike
            elif (kama_falling and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit on opposite KAMA direction or Chop > 61.8 (range)
            if position == 1:
                if (not kama_rising or chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (not kama_falling or chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Chop_VolumeSpike"
timeframe = "4h"
leverage = 1.0