#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Regime_ATRStop_v1
Hypothesis: Donchian(20) breakout with volume confirmation and choppiness regime filter on 4h timeframe.
Long when price breaks above upper Donchian channel with volume > 1.5x average and CHOP > 61.8 (ranging market).
Short when price breaks below lower Donchian channel with volume > 1.5x average and CHOP > 61.8.
Uses ATR-based trailing stop (2.0x ATR from extreme) to manage risk.
Designed for low trade frequency (20-50/year) to avoid fee drag while capturing breakouts in ranging markets.
Uses discrete position sizing (0.25) to minimize fee churn.
Works in both bull and bear markets by fading breakouts in choppy regimes.
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
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low))) / log10(n)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(sum_atr_14 / (14 * np.log10(range_14))) / np.log10(14)
    # Handle invalid values
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of Donchian (20), ATR (14), volume MA (20), CHOP (14)
    start_idx = max(20, 14, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        hh = highest_high[i]
        ll = lowest_low[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        chop_val = chop[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian, volume confirmation, choppy market (mean reversion)
            long_signal = (close_val > hh) and (volume_val > 1.5 * vol_ma_val) and (chop_val > 61.8)
            # Short: price breaks below lower Donchian, volume confirmation, choppy market (mean reversion)
            short_signal = (close_val < ll) and (volume_val > 1.5 * vol_ma_val) and (chop_val > 61.8)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_val)
            # Exit: trailing stop hit or price re-enters Donchian channel
            if (low_val < long_stop) or (close_val < hh):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_val)
            # Exit: trailing stop hit or price re-enters Donchian channel
            if (high_val > short_stop) or (close_val > ll):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Regime_ATRStop_v1"
timeframe = "4h"
leverage = 1.0