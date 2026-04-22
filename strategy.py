#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with volume spike and 1d Choppiness regime filter.
# Uses 1d Choppiness Index to identify trending vs ranging markets.
# Long when price > KAMA(10) in trending market with volume spike.
# Short when price < KAMA(10) in trending market with volume spike.
# Designed for 4h timeframe to target 20-50 trades/year per symbol.
# Works in bull/bear via regime filtering and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14-period)
    # ATR(14) = sum of true ranges over 14 periods
    # sum_high_low = sum of (high - low) over 14 periods
    # CHOP = 100 * log10(sum_true_range / (max_high - min_low)) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Handle first value for true range
    tr[0] = high_1d[0] - low_1d[0]
    
    # Calculate ATR(14) and sum of high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hl_range = high_1d - low_1d
    sum_hl_14 = pd.Series(hl_range).rolling(window=14, min_periods=14).sum().values
    
    # Calculate max(high) and min(low) over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    mask = (sum_hl_14 > 0) & (max_high_14 > min_low_14)
    chop[mask] = 100 * np.log10(atr_14[mask] * 14 / sum_hl_14[mask]) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h KAMA(10) for trend
    # Efficiency Ratio = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = pd.Series(change).rolling(window=10, min_periods=10).sum().values / \
         pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike filter (20-period on 4h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(kama[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in trending markets (CHOP < 38.2)
        if chop_aligned[i] < 38.2:
            if position == 0:
                # Long: price above KAMA + volume spike
                if close[i] > kama[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA + volume spike
                elif close[i] < kama[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
                # Exit on trend reversal
                if (position == 1 and close[i] < kama[i]) or \
                   (position == -1 and close[i] > kama[i]):
                    signals[i] = 0.0
                    position = 0
        else:
            # In ranging markets, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0