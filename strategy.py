#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX with 1d volume spike and choppiness regime filter.
- TRIX(12): Triple Exponential Average, momentum oscillator
- Long: TRIX crosses above zero (bullish momentum) + 1d volume > 2.0x 20-period average + choppiness regime > 50 (range/momentum transition)
- Short: TRIX crosses below zero (bearish momentum) + 1d volume > 2.0x 20-period average + choppiness regime > 50
- Exit: TRIX crosses back through zero in opposite direction OR choppiness drops below 30 (strong trend - avoid whipsaw)
- Volume spike ensures institutional participation
- Choppiness filter avoids whipsaw in strong trends while allowing momentum captures in transitional markets
- Works in bull markets (momentum continuation) and bear markets (mean reversion via momentum exhaustion)
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average (institutional participation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX(12) on 12h data
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.full_like(close, np.nan)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # Percentage change
    
    # 1d EMA34 for trend context (not direct signal, but for regime)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (n * (HHV - LLV))) / log10(n)
    # Simplified: CHOP > 50 = ranging, CHOP < 30 = trending
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # First TR is just high-low
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        
        hhvs = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        llvs = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        hhvs[0:period-1] = np.nan
        llvs[0:period-1] = np.nan
        
        chop = np.full_like(close_arr, np.nan)
        valid = (atr > 0) & (hhvs > llvs) & ~np.isnan(hhvs) & ~np.isnan(llvs)
        chop[valid] = 100 * np.log10(np.sum(atr[valid:valid+period]) / (period * (hhvs[valid] - llvs[valid]))) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 12*3, 34, 14)  # Volume MA, TRIX (3x EMA12), EMA34, CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        # Choppiness regime: > 50 = ranging/transitional (good for momentum), < 30 = strong trend (avoid)
        chop_regime = chop_aligned[i] > 50
        
        # TRIX zero crossover signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + chop regime > 50
            if volume_spike and chop_regime and trix_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume spike + chop regime > 50
            elif volume_spike and chop_regime and trix_cross_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR chop drops below 30 (strong trend - avoid whipsaw)
            if trix_cross_down or chop_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR chop drops below 30 (strong trend - avoid whipsaw)
            if trix_cross_up or chop_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0