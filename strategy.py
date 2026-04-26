#!/usr/bin/env python3
"""
4h_RSI_Chop_Donchian20_Breakout
Hypothesis: Combine RSI mean reversion with Donchian breakout and choppiness regime filter.
Long when: RSI<30 + price breaks above Donchian(20) high + CHOP>61.8 (range regime).
Short when: RSI>70 + price breaks below Donchian(20) low + CHOP>61.8 (range regime).
Exit when: price reverts to Donchian midpoint or regime shifts to trending (CHOP<38.2).
Uses 4h timeframe with discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- Works in ranging markets (2025+ bear/range) via mean reversion at extremes
- Volume not required; relies on price action and regime
- Targets 20-40 trades/year for optimal test generalization.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Donchian(20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Choppiness Index(14)
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    atr = pd.Series(atr_list).rolling(window=14, min_periods=14).mean().values
    
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    chop = 100 * np.log10(np.sum(atr, axis=0) / np.log10(14) / hl_range) if False else \
           100 * np.log10(np.nansum(atr) / np.log10(14) / hl_range) if False else \
           100 * np.log10(pd.Series(atr).rolling(14, min_periods=14).sum().values / np.log10(14) / hl_range)
    # Simplified: chop = 100 * log10(sum(atr(14)) / log10(14) / (HH-LL))
    atr_sum = pd.Series(atr_list).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(14) / hl_range)
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 14 for RSI/CHOP
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(chop[i]) or np.isnan(donch_mid[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry in ranging regime (CHOP > 61.8)
            if chop[i] > 61.8:
                # Long: RSI oversold + break above Donchian high
                long_entry = (rsi[i] < 30) and (close_val > donch_high[i])
                # Short: RSI overbought + break below Donchian low
                short_entry = (rsi[i] > 70) and (close_val < donch_low[i])
                
                if long_entry:
                    signals[i] = size
                    position = 1
                elif short_entry:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to midpoint or regime shifts to trending
            if close_val < donch_mid[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint or regime shifts to trending
            if close_val > donch_mid[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI_Chop_Donchian20_Breakout"
timeframe = "4h"
leverage = 1.0