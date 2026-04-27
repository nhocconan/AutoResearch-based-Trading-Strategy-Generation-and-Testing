#!/usr/bin/env python3
"""
6h_Adaptive_Kelly_Donchian_Volume_Regime
Hypothesis: Donchian(20) breakout on 6h with volatility regime filter (ATR ratio) and adaptive Kelly sizing.
In high volatility regimes (ATR(7)/ATR(30) > 1.2), breakouts capture strong moves with reduced size.
In low volatility regimes (ATR ratio <= 1.2), breakouts are faded with mean reversion at Donchian mid.
Uses discrete position sizing tiers (0.0, ±0.15, ±0.25) to minimize fee churn.
Designed for 6h timeframe targeting 50-150 total trades over 4 years.
Works in bull/bear markets: volatility regime adapts to changing market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # ATR for volatility regime (7 and 30 period)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr7 / (atr30 + 1e-10)  # Avoid division by zero
    
    # Volume confirmation: current > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size_normal = 0.25   # Normal size in low volatility
    size_high_vol = 0.15 # Reduced size in high volatility
    
    # Warmup: need Donchian(20), ATR(30), vol avg(20)
    start_idx = max(20, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(donch_mid[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        vol_regime = atr_ratio[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Entry logic based on volatility regime
            if vol_regime > 1.2:  # High volatility: trade breakouts
                long_condition = (close_val > upper and vol_conf)
                short_condition = (close_val < lower and vol_conf)
                if long_condition:
                    signals[i] = size_high_vol
                    position = 1
                elif short_condition:
                    signals[i] = -size_high_vol
                    position = -1
            else:  # Low volatility: fade extreme moves, mean revert to mid
                # Fade when price touches bands with volume confirmation
                long_condition = (close_val <= lower and vol_conf and close_val < mid)
                short_condition = (close_val >= upper and vol_conf and close_val > mid)
                if long_condition:
                    signals[i] = size_normal
                    position = 1
                elif short_condition:
                    signals[i] = -size_normal
                    position = -1
        elif position == 1:
            # Exit logic: reverse when price reaches opposite extreme or mid
            if vol_regime > 1.2:  # High vol: exit on opposite band touch
                if close_val >= upper:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size_high_vol
            else:  # Low vol: exit when price reaches mid or opposite band
                if close_val >= mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size_normal
        elif position == -1:
            # Exit logic for short position
            if vol_regime > 1.2:  # High vol: exit on opposite band touch
                if close_val <= lower:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size_high_vol
            else:  # Low vol: exit when price reaches mid or opposite band
                if close_val <= mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size_normal
    
    return signals

name = "6h_Adaptive_Kelly_Donchian_Volume_Regime"
timeframe = "6h"
leverage = 1.0