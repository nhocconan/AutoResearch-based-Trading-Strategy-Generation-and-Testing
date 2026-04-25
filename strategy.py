#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm
Hypothesis: 12h Donchian(20) breakout with 1w Camarilla pivot trend filter and volume confirmation (>1.5x 20-bar mean volume). Long when price > upper Donchian(20) and above weekly Camarilla H3 pivot; short when price < lower Donchian(20) and below weekly Camarilla L3 pivot. Uses discrete position sizing (0.25) to minimize fee drag. Designed for 12-30 trades/year per symbol, effective in both bull (breakouts with volume) and bear (trend-following via shorts) markets by aligning with higher-timeframe structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (Camarilla pivots)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Camarilla levels (H3 and L3) from previous 1w bar
    camarilla_h3_1w = close_1w + 1.1 * (high_1w - low_1w)  # H3 = C + 1.1*(H-L)
    camarilla_l3_1w = close_1w - 1.1 * (high_1w - low_1w)  # L3 = C - 1.1*(H-L)
    
    # Align weekly Camarilla levels to 12h timeframe (use previous bar's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # Calculate Donchian(20) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume mean
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high and above weekly H3 with volume confirmation
            long_signal = (close[i] > donchian_high[i]) and (close[i] > camarilla_h3_aligned[i]) and vol_confirm[i]
            # Short: price breaks below Donchian(20) low and below weekly L3 with volume confirmation
            short_signal = (close[i] < donchian_low[i]) and (close[i] < camarilla_l3_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian(20) low (trend reversal)
            exit_signal = close[i] < donchian_low[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian(20) high (trend reversal)
            exit_signal = close[i] > donchian_high[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0