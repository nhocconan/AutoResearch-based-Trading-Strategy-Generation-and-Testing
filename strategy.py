#!/usr/bin/env python3
"""
Experiment #2947: 6h Elder Ray + ADX Regime Filter v2
HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure.
ADX(14) > 25 filters for trending markets only, avoiding chop. Long when Bull Power > 0 and ADX trending;
Short when Bear Power < 0 and ADX trending. Uses 6h timeframe for medium-term signals with
discrete position sizing (0.25) to manage drawdown. Volume confirmation (>1.5x average) ensures
breakout validity. Designed to work in both bull (2021, 2023-2024) and bear (2022, 2025+) regimes
by requiring trend confirmation via ADX.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2947_6h_elder_ray_adx_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 6h Indicators: EMA(13) for Elder Ray calculation ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h Indicators: Elder Ray Components ===
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # === 6h Indicators: ADX(14) for trend strength ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR and DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.mean(data[0:period])  # Seed with SMA
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_loss = 0.0
    
    warmup = 50  # sufficient for all indicators (EMA13, ADX14, VolMA20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if ADX weakens (< 20) - trend ending
            if adx[i] < 20.0:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Exit on opposing Elder Ray signal
            if position_side > 0:  # Long
                if bear_power[i] >= 0:  # Bear Power turned positive (exit long)
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if bull_power[i] <= 0:  # Bull Power turned negative (exit short)
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_ok = vol_ratio[i] > 1.5
        
        # Require trending market (ADX > 25)
        trending = adx[i] > 25.0
        
        if volume_ok and trending:
            # Long entry: Bull Power positive (buying pressure) + trending
            if bull_power[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                # Initial stop loss at 2*ATR below entry
                stop_loss = entry_price - 2.0 * atr[i]
                signals[i] = SIZE
            # Short entry: Bear Power negative (selling pressure) + trending
            elif bear_power[i] < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                # Initial stop loss at 2*ATR above entry
                stop_loss = entry_price + 2.0 * atr[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>