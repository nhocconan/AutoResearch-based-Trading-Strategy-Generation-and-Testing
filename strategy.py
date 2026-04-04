#!/usr/bin/env python3
"""
Experiment #2927: 6h Elder Ray + ADX Regime Filter (Novel Adaptive Version)
HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure.
ADX(14) > 25 filters for trending markets only, preventing whipsaw in ranges.
Unlike fixed-threshold strategies, this uses dynamic regime detection:
- In strong trends (ADX>25): trade Elder Ray signals in trend direction
- In weak trends/ranges (ADX<=25): fade Elder Ray extremes (mean reversion)
This adaptive approach works in both bull (2021, 2023-24) and bear (2022, 2025+) regimes.
6h timeframe provides sufficient signal quality while keeping trades ~20-40/year.
Target: 80-180 total trades over 4 years (20-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2927_6h_elder_ray_adx_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === EMA(13) for Elder Ray baseline (13-period EMA) ===
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === Elder Ray Components ===
    bull_power = high - ema13          # Buying power: high vs EMA13
    bear_power = low - ema13           # Selling power: low vs EMA13 (negative)
    
    # === ADX(14) Calculation ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing: EMA with alpha=1/period"""
        return pd.Series(data).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Sufficient for all indicators
    
    for i in range(warmup, n):
        # Skip if any indicator is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(ema13[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Fixed stop at 2.5*ATR ---
        if in_position:
            atr_now = atr[i]
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * atr_now:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.5 * atr_now:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- Adaptive Entry Logic Based on Regime ---
        adx_val = adx[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if adx_val > 25:  # Trending regime: follow Elder Ray
            # Strong buying pressure + bullish bias = long
            if bull_val > 0 and ema13[i] > ema13[i-1]:
                in_position = True
                position_side = 1
                entry_price = price
                signals[i] = SIZE
            # Strong selling pressure + bearish bias = short
            elif bear_val < 0 and ema13[i] < ema13[i-1]:
                in_position = True
                position_side = -1
                entry_price = price
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:  # Ranging regime: fade Elder Ray extremes
            # Extreme buying pressure = short (expect pullback)
            if bull_val > np.percentile(bull_power[max(0,i-100):i+1], 85):
                in_position = True
                position_side = -1
                entry_price = price
                signals[i] = -SIZE
            # Extreme selling pressure = long (expect bounce)
            elif bear_val < np.percentile(bear_power[max(0,i-100):i+1], 15):
                in_position = True
                position_side = 1
                entry_price = price
                signals[i] = SIZE
            else:
                signals[i] = 0.0
    
    return signals