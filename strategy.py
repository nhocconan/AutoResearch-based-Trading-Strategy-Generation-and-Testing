#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_Chop
Hypothesis: 4h Donchian(20) breakout with volume confirmation (>2.0x 20-bar mean) and choppiness regime filter (CHOP(14) > 61.8 = range for mean reversion, CHOP < 38.2 = trend for trend following). In ranging markets (CHOP>61.8): fade Donchian breaks (short at upper band, long at lower band). In trending markets (CHOP<38.2): follow Donchian breaks (long at upper band, short at lower band). Uses discrete position sizing (0.25) to minimize fee churn. Designed for 20-40 trades/year per symbol, effective in both bull (breakouts with volume) and bear (mean reversion in ranges) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    # Choppiness Index (CHOP) - 14 period
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # ATR-like component: sum of true ranges over 14 periods
    atr_sum = tr_sum
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_sum / range_min_min) / np.log10(14)
    # Handle invalid values
    chop = np.where((range_max_min > 0) & (atr_sum > 0), chop_raw, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian, volume, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_mean_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Determine regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (follow)
            is_ranging = chop[i] > 61.8
            is_trending = chop[i] < 38.2
            
            # In ranging markets: mean reversion at Donchian bands
            # In trending markets: follow breakouts
            
            if is_ranging:
                # Fade Donchian breaks: short at upper band, long at lower band
                long_signal = (close[i] < lowest_20[i]) and vol_confirm[i]
                short_signal = (close[i] > highest_20[i]) and vol_confirm[i]
            elif is_trending:
                # Follow Donchian breaks: long at upper band, short at lower band
                long_signal = (close[i] > highest_20[i]) and vol_confirm[i]
                short_signal = (close[i] < lowest_20[i]) and vol_confirm[i]
            else:
                # Choppy transition zone: no signals
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below midpoint (mean reversion) or opposite band (trend exhaustion)
            midpoint = (highest_20[i] + lowest_20[i]) / 2
            exit_signal = close[i] < midpoint
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above midpoint
            midpoint = (highest_20[i] + lowest_20[i]) / 2
            exit_signal = close[i] > midpoint
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_Chop"
timeframe = "4h"
leverage = 1.0