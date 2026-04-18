#!/usr/bin/env python3
"""
12h_1D_Choppiness_Regime_Donchian_Breakout
Hypothesis: On 12h timeframe, trade breakouts of Donchian(20) channels filtered by daily Choppiness Index regime.
In trending regimes (CHOP < 38.2), trade breakouts in direction of trend (price > SMA50 for long, < SMA50 for short).
In ranging regimes (CHOP > 61.8), fade extremes at Donchian bands with smaller size.
Uses volume confirmation to avoid false breakouts. Position sizing 0.25 for trend breakouts, 0.15 for mean reversion.
Designed for low trade frequency (~20-40/year) to minimize fee drag while capturing trends and range reversals.
Works in bull markets via trend breaks and in bear/ranging via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # SMA50 for trend filter
    sma50 = np.full(n, np.nan)
    for i in range(50, n):
        sma50[i] = np.mean(close[i-50:i])
    
    # Volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Get 1D data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1D
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    
    # ATR(14) for Choppiness denominator
    atr = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    atr_sum = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr_sum[i] = np.sum(atr[0:14])
        else:
            atr_sum[i] = atr_sum[i-1] - atr[i-14] + atr[i]
    
    # Max(high) - Min(low) over 14 periods
    max_high = np.full(len(close_1d), np.nan)
    min_low = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        max_high[i] = np.max(high_1d[i-14:i])
        min_low[i] = np.min(low_1d[i-14:i])
    
    # Choppiness Index: 100 * log10(atr_sum / (max_high - min_low)) / log10(14)
    chop = np.full(len(close_1d), 50.0)  # default to middle
    for i in range(14, len(close_1d)):
        if max_high[i] > min_low[i] and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Align Choppiness to 12h timeframe (wait for bar close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need SMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(sma50[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Determine regime
            if chop_aligned[i] < 38.2:  # Trending regime
                # Long breakout: price > Donchian high + volume + above SMA50
                if close[i] > highest_high[i] and vol_confirm and close[i] > sma50[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price < Donchian low + volume + below SMA50
                elif close[i] < lowest_low[i] and vol_confirm and close[i] < sma50[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop_aligned[i] > 61.8:  # Ranging regime
                # Fade at Donchian bands with smaller size
                if close[i] > highest_high[i] and vol_confirm:
                    signals[i] = -0.15  # Short at upper band
                    position = -1
                elif close[i] < lowest_low[i] and vol_confirm:
                    signals[i] = 0.15   # Long at lower band
                    position = 1
        
        elif position == 1:
            # Long exit: price < Donchian low OR reverse signal in ranging
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] > 61.8 and close[i] < sma50[i]:  # Exit long in range if below SMA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > Donchian high OR reverse signal in ranging
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] > 61.8 and close[i] > sma50[i]:  # Exit short in range if above SMA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.15
    
    return signals

name = "12h_1D_Choppiness_Regime_Donchian_Breakout"
timeframe = "12h"
leverage = 1.0