#!/usr/bin/env python3
"""
6h_Alligator_ElderRay_Regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) combined with Williams Alligator for regime filtering.
- Williams Alligator (jaw/teeth/lips) defines trend regime: bullish when lips > teeth > jaw, bearish when lips < teeth < jaw.
- Elder Ray: Bull Power = High - EMA13(Teeth), Bear Power = EMA13(Teeth) - Low.
- Entry: In bull regime, go long when Bull Power > 0 and rising (2-bar momentum). In bear regime, go short when Bear Power > 0 and rising.
- Exit: Opposite signal or Alligator regime change (teeth crosses lips).
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year) by requiring Alligator alignment + Elder Ray momentum.
- Discrete sizing (0.25) minimizes fee churn. ATR-based stop (2.0x) controls risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 6h Williams Alligator (13,8,5 SMAs shifted) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # === 6h Elder Ray components ===
    # EMA13 of close (same as teeth base but unshifted for power calculation)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Momentum of power (2-bar change)
    bull_power_mom = bull_power - np.roll(bull_power, 2)
    bear_power_mom = bear_power - np.roll(bear_power, 2)
    bull_power_mom[:2] = np.nan
    bear_power_mom[:2] = np.nan
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 24  # max 6 days (24 * 6h = 144h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(atr[i]) or np.isnan(bull_power_mom[i]) or np.isnan(bear_power_mom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        
        # Alligator regime
        is_bull_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        is_bear_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if position == 0:
            if is_bull_alligator:
                # Bull regime: long when Bull Power positive and rising
                long_condition = (bull_power[i] > 0) and (bull_power_mom[i] > 0)
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
            elif is_bear_alligator:
                # Bear regime: short when Bear Power positive and rising
                short_condition = (bear_power[i] > 0) and (bear_power_mom[i] > 0)
                if short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit on regime change or power fading
                elif not is_bull_alligator or (bull_power[i] <= 0) or (bull_power_mom[i] < 0):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit on regime change or power fading
                elif not is_bear_alligator or (bear_power[i] <= 0) or (bear_power_mom[i] < 0):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Alligator_ElderRay_Regime_v1"
timeframe = "6h"
leverage = 1.0