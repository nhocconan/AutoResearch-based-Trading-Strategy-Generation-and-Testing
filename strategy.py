#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian breakout with volume confirmation.
# Uses Choppiness Index from 4h data to determine market regime: 
# - CHOP > 61.8 = ranging market (mean reversion): fade Donchian breakouts
# - CHOP < 38.2 = trending market (trend following): trade Donchian breakouts
# This regime filter adapts to both bull and bear markets by focusing on trend strength rather than direction.
# Combined with volume confirmation to avoid false breakouts.
# Target: 20-50 trades/year to avoid fee drag.
name = "4h_ChopRegime_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (4h) - regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # High CHOP (>61.8) = ranging, Low CHOP (<38.2) = trending
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # Align with index
    
    atr1 = tr1  # ATR(1) is just true range
    atr10 = pd.Series(atr1).rolling(window=10, min_periods=10).mean().values
    
    # Sum of true ranges over 10 periods
    sum_tr10 = pd.Series(tr1).rolling(window=10, min_periods=10).sum().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr10 / (10 * atr10)) / np.log10(10)
    chop = np.where(atr10 > 0, chop, 50)  # Handle division by zero
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(chop[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Only trade in trending markets (CHOP < 38.2)
            if chop[i] < 38.2 and vol_confirm[i]:
                # Enter long: price breaks above Donchian high
                if price > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Enter short: price breaks below Donchian low
                elif price < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian low or chop increases (range developing)
            if price < lowest_low[i] or chop[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian high or chop increases (range developing)
            if price > highest_high[i] or chop[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals