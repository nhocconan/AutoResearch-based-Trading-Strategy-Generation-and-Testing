#!/usr/bin/env python3
"""
12h_KAMA_Regime_DonchianBreakout_v1
Hypothesis: 12h KAMA trend + Donchian(20) breakout + volume confirmation + chop regime filter.
- KAMA (adaptive trend) from 12h data determines trend direction
- Donchian(20) breakout in direction of KAMA trend for entry
- Volume confirmation: current volume > 1.5 * 20-period average volume
- Choppiness regime filter: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion edge
- Works in both bull/bear markets by using adaptive trend (KAMA) and mean reversion in ranging conditions
- Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
- Uses 12h primary timeframe with 1d HTF for regime filtering (optional extension)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAMA Calculation (10, 2, 30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will fix with loop below
    # Recalculate volatility properly
    volatility = np.zeros_like(change)
    for i in range(len(change)):
        volatility[i] = np.sum(np.abs(np.diff(close[i:i+10])))
    er = np.zeros_like(change, dtype=np.float64)
    er[volatility != 0] = change[volatility != 0] / volatility[volatility != 0]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === Donchian Channels (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Average (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Choppiness Index (14) ===
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/Min High-Low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_high - min_low)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = max_high - min_low
    chop = np.full_like(close, np.nan, dtype=np.float64)
    mask = (denominator > 0) & (~np.isnan(sum_tr_14)) & (~np.isnan(denominator))
    chop[mask] = 100 * np.log10(sum_tr_14[mask] / denominator[mask]) / np.log10(14)
    
    # === Align Indicators (already on 12h timeframe) ===
    # No MTF alignment needed for primary timeframe indicators
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 30 for KAMA, 14 for Chop)
    start_idx = max(20, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Conditions
        price_above_donchian = close[i] > donchian_high[i]
        price_below_donchian = close[i] < donchian_low[i]
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        chop_high = chop[i] > 61.8  # Ranging market
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        if position == 0:
            # Long: price breaks above Donchian AND volume confirm AND chop high AND price > KAMA
            if price_above_donchian and volume_confirm and chop_high and kama_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian AND volume confirm AND chop high AND price < KAMA
            elif price_below_donchian and volume_confirm and chop_high and kama_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR volume drops OR chop low (trending)
            if price_below_donchian or not volume_confirm or chop[i] <= 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR volume drops OR chop low (trending)
            if price_above_donchian or not volume_confirm or chop[i] <= 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Regime_DonchianBreakout_v1"
timeframe = "12h"
leverage = 1.0