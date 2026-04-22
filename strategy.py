#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Choppiness Index regime filter + Donchian breakout + volume confirmation
    # Choppiness Index identifies trending vs ranging markets (trend when CHOP < 38.2, range when > 61.8)
    # In trending regimes: trade Donchian breakouts with volume confirmation
    # In ranging regimes: fade Donchian breakouts (mean reversion)
    # Works in bull/bear: adapts to market regime
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((atr_safe * atr_period) / (highest_high - lowest_low)) / np.log10(atr_period)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Determine market regime
            if chop_val < 38.2:  # Trending regime
                # Long: Donchian breakout up with volume
                if close[i] > donchian_high[i] and vol_ok:
                    signals[i] = 0.30
                    position = 1
                # Short: Donchian breakout down with volume
                elif close[i] < donchian_low[i] and vol_ok:
                    signals[i] = -0.30
                    position = -1
            elif chop_val > 61.8:  # Ranging regime
                # Long: Fade Donchian breakdown (mean reversion)
                if close[i] < donchian_low[i] and vol_ok:
                    signals[i] = 0.30
                    position = 1
                # Short: Fade Donchian breakout up (mean reversion)
                elif close[i] > donchian_high[i] and vol_ok:
                    signals[i] = -0.30
                    position = -1
            # Chop between 38.2-61.8: transition, no trades
        else:
            # Exit conditions
            if position == 1:  # Long position
                if chop_val < 38.2:  # Still trending
                    # Exit on Donchian breakdown
                    if close[i] < donchian_low[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.30
                else:  # Ranging or transition
                    # Exit on mean reversion to midpoint
                    midpoint = (donchian_high[i] + donchian_low[i]) / 2
                    if close[i] > midpoint:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.30
            else:  # position == -1 (Short)
                if chop_val < 38.2:  # Still trending
                    # Exit on Donchian breakout up
                    if close[i] > donchian_high[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.30
                else:  # Ranging or transition
                    # Exit on mean reversion to midpoint
                    midpoint = (donchian_high[i] + donchian_low[i]) / 2
                    if close[i] < midpoint:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.30
    
    return signals

name = "4h_Choppiness_Donchian_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0