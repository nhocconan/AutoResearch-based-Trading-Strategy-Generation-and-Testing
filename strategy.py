#!/usr/bin/env python3
"""
4h_RSI_Regime_Donchian_Breakout
Hypothesis: Use RSI(14) for momentum, Donchian(20) for breakout direction, and Choppiness Index for regime filtering.
Enter long when price breaks above Donchian high, RSI > 50, and market is trending (CHOP < 38.2).
Enter short when price breaks below Donchian low, RSI < 50, and market is trending (CHOP < 38.2).
Exit when RSI crosses back to 50 or volatility spikes (ATR ratio > 2.5). Works in bull/bear via RSI and regime filters.
"""

name = "4h_RSI_Regime_Donchian_Breakout"
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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Donchian(20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14) - uses high/low/close
    atr1 = np.maximum(np.abs(high - low), np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr1[0] = np.abs(high[0] - low[0])
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.divide(100 * np.log10(atr_sum) / np.log10(14), 
                     np.log10((highest_high - lowest_low)), 
                     out=np.zeros_like(atr_sum), 
                     where=(highest_high - lowest_low)!=0)
    
    # ATR(20) for volatility filter
    tr1 = np.maximum(np.abs(high - low), np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = np.abs(high[0] - low[0])
    atr = pd.Series(tr1).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.divide(atr, atr_ma, out=np.ones_like(atr), where=atr_ma!=0)
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(chop[i]) or np.isnan(atr_ratio[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: trending market (CHOP < 38.2)
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Long: break above Donchian high, RSI > 50, trending market, volume spike
            if (close[i] > donch_high[i] and
                rsi[i] > 50 and
                is_trending and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, RSI < 50, trending market, volume spike
            elif (close[i] < donch_low[i] and
                  rsi[i] < 50 and
                  is_trending and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI < 50 or volatility spike
            if rsi[i] < 50 or atr_ratio[i] > 2.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI > 50 or volatility spike
            if rsi[i] > 50 or atr_ratio[i] > 2.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals