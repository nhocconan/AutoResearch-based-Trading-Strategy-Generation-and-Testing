#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Chop > 61.8 = range (mean revert at Donchian edges), Chop < 38.2 = trending (breakout follow)
# Works in bull/bear by adapting to market regime. Target: 20-40 trades/year via regime filter.
# Entry: Long when Chop < 38.2 (trending) + price > Donchian high(20) + volume > 1.5x avg
# Short when Chop < 38.2 + price < Donchian low(20) + volume > 1.5x avg
# Exit: Opposite Donchian touch OR Chop > 61.8 (range) triggers mean reversion exit

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    atr_d = np.zeros(len(close_d))
    tr = np.maximum(high_d[1:] - low_d[1:], 
                    np.abs(high_d[1:] - close_d[:-1]),
                    np.abs(low_d[1:] - close_d[:-1]))
    atr_d[1:] = tr
    # Wilder's smoothing for ATR
    atr_smoothed = np.zeros_like(atr_d)
    if len(atr_d) > 14:
        atr_smoothed[14] = np.mean(atr_d[1:15])
        for i in range(15, len(atr_d)):
            atr_smoothed[i] = (atr_smoothed[i-1] * 13 + atr_d[i]) / 14
    # Sum of ATR over 14 periods
    sum_atr = np.zeros_like(close_d)
    for i in range(14, len(close_d)):
        sum_atr[i] = np.sum(atr_smoothed[i-13:i+1])
    # Chop = 100 * log10(sumATR / (maxHH - minLL)) / log10(14)
    max_hh = np.zeros_like(close_d)
    min_ll = np.zeros_like(close_d)
    for i in range(14, len(close_d)):
        max_hh[i] = np.max(high_d[i-13:i+1])
        min_ll[i] = np.min(low_d[i-13:i+1])
    chop = np.full_like(close_d, 50.0, dtype=float)
    for i in range(14, len(close_d)):
        if max_hh[i] != min_ll[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_hh[i] - min_ll[i])) / np.log10(14)
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Load 4h data for volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    vol_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        vol_current = vol_4h[i]  # Already 4h aligned
        
        # Regime filters
        trending = chop_val < 38.2
        ranging = chop_val > 61.8
        
        # Volume confirmation
        volume_confirm = vol_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Enter only in trending regime with volume
            if trending and volume_confirm:
                # Long breakout
                if prices['close'].iloc[i] > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown
                elif prices['close'].iloc[i] < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: opposite Donchian touch OR market goes ranging
                if prices['close'].iloc[i] < donchian_low[i]:
                    exit_signal = True
                elif ranging:  # Mean reversion exit in range
                    exit_signal = True
            elif position == -1:
                # Exit short: opposite Donchian touch OR market goes ranging
                if prices['close'].iloc[i] > donchian_high[i]:
                    exit_signal = True
                elif ranging:  # Mean reversion exit in range
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Chop_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0