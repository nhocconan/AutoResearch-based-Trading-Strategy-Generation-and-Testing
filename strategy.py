#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_v1
# Hypothesis: Daily Donchian(20) breakout with volume confirmation and choppiness regime filter.
# In ranging markets (2025+), price tends to revert from channel extremes. 
# Volume breakout filters false signals. Chop filter ensures we only trade in ranging regimes.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 50-100 total trades over 4 years.
# Primary timeframe: 1d, HTF: 1w for regime filter (optional, but can add if needed).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index regime filter (14-period)
    high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 50)
        chop_regime = chop[i] > 50
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or volume dries up
            if close[i] < donchian_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or volume dries up
            if close[i] > donchian_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above Donchian high with volume
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low with volume
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals