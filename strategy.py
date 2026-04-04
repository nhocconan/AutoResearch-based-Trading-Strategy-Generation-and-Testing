#!/usr/bin/env python3
"""
Experiment #2915: 6h Elder Ray + ADX Regime Filter
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) measures trend strength relative to EMA13.
ADX > 25 filters for trending markets. Long when Bull Power > 0 and ADX rising.
Short when Bear Power < 0 and ADX rising. 6h timeframe reduces noise vs lower TFs.
Target: 75-150 total trades over 4 years (19-37/year). Discrete sizing: 0.25.
Works in both bull (strong Bull Power) and bear (strong Bear Power) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2915_6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # === Indicators: ADX(14) for trend strength ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(x, period):
        return pd.Series(x).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # Skip if any indicator is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(di_plus[i]) or np.isnan(di_minus[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic ---
        if in_position:
            # Exit if ADX weakens (< 20) or power reverses
            if position_side > 0:  # Long
                if adx[i] < 20.0 or bull_power[i] <= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if adx[i] < 20.0 or bear_power[i] >= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require ADX > 25 for trending market
        if adx[i] > 25.0:
            # Long: Bull Power positive AND DI+ > DI- (bullish momentum)
            if bull_power[i] > 0 and di_plus[i] > di_minus[i]:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short: Bear Power negative AND DI- > DI+ (bearish momentum)
            elif bear_power[i] < 0 and di_minus[i] > di_plus[i]:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals