#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian(20) breakout and volume confirmation.
# Uses Choppiness Index (26-period) to detect trending vs ranging markets:
#   CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow).
# In trending regimes: take Donchian breakout with volume confirmation.
# In ranging regimes: fade Donchian breakout (mean reversion) with volume confirmation.
# Volume confirmation: volume > 1.5x 24-period moving average.
# Designed for 4h timeframe to capture trend and range regimes with low frequency.
# Target: 20-40 trades/year per symbol (80-160 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (26-period) on close prices
    # CHOP = 100 * log10(sum(ATR(1)) / (highesthigh - lowestlow)) / log10(period)
    tr1 = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with original length
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    
    sum_atr1 = pd.Series(atr1).rolling(window=26, min_periods=26).sum().values
    highest_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    lowest_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 100 * np.log10(sum_atr1 / range_hl) / np.log10(26), 50)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Regime thresholds
    chop_ranging = chop > 61.8  # ranging market (mean revert)
    chop_trending = chop < 38.2  # trending market (trend follow)
    
    # Donchian channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (24-period average)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > 1.5 * vol_ma24
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending regime: follow breakout
            if chop_trending[i]:
                # Long: breakout above Donchian high with volume
                if close[i] > highest_high_20[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: breakdown below Donchian low with volume
                elif close[i] < lowest_low_20[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime: fade breakout (mean reversion)
            elif chop_ranging[i]:
                # Short: fade breakout above Donchian high (sell the spike)
                if close[i] > highest_high_20[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                # Long: fade breakdown below Donchian low (buy the dip)
                elif close[i] < lowest_low_20[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
        else:
            # Exit conditions: opposite signal or regime change
            if position == 1:
                # Exit long: breakdown below Donchian low or shift to ranging regime
                if close[i] < lowest_low_20[i] or chop_ranging[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: breakout above Donchian high or shift to ranging regime
                if close[i] > highest_high_20[i] or chop_ranging[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Regime_Donchian20_Breakout_Volume"
timeframe = "4h"
leverage = 1.0