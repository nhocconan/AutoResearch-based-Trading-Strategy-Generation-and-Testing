#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(9) + volume spike + choppiness regime filter
# Long when TRIX crosses above zero AND chop > 61.8 (ranging) AND volume > 2.0x 20-period average
# Short when TRIX crosses below zero AND chop > 61.8 (ranging) AND volume > 2.0x 20-period average
# Exit when TRIX crosses zero in opposite direction OR chop < 38.2 (trending) OR volume normalizes
# TRIX is a momentum oscillator that filters noise and identifies trend changes
# Choppiness regime filter ensures we only trade in ranging markets where mean reversion works
# Volume spike confirms institutional participation in the move
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 12h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "12h_TRIX_Chop_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(9) on close
    # TRIX = EMA(EMA(EMA(close, 9), 9), 9) - 1 period ago
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # first value has no previous
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(1), 14) / (log10(highest_high - lowest_low) * log10(14)))
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0)  # default to neutral
    mask = (range_hl > 0) & ~np.isnan(range_hl) & ~np.isnan(atr1)
    chop[mask] = 100 * np.log10(np.sum(atr1) / (np.log10(range_hl[mask]) * np.log10(14))) if np.sum(atr1) > 0 else 50.0
    # Recalculate properly for each bar
    for i in range(14, n):
        if range_hl[i] > 0 and not np.isnan(range_hl[i]) and not np.isnan(atr1[i]):
            atr_sum = np.nansum(atr1[i-13:i+1]) if i >= 13 else np.nansum(atr1[:i+1])
            chop[i] = 100 * np.log10(atr_sum / (np.log10(range_hl[i]) * np.log10(14)))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: TRIX crosses above zero AND chop > 61.8 (ranging) AND volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                chop[i] > 61.8 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX crosses below zero AND chop > 61.8 (ranging) AND volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  chop[i] > 61.8 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero OR chop < 38.2 (trending) OR volume normalizes
            if (trix[i] < 0 and trix[i-1] >= 0) or \
               chop[i] < 38.2 or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero OR chop < 38.2 (trending) OR volume normalizes
            if (trix[i] > 0 and trix[i-1] <= 0) or \
               chop[i] < 38.2 or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals