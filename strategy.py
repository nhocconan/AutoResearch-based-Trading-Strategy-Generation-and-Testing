#!/usr/bin/env python3
# 6h_donchian_breakout_weekly_pivot_volume_v1
# Hypothesis: 6h strategy combining Donchian(20) breakout with weekly Camarilla pivot levels (R4/S4) for breakout confirmation, volume filter (>1.3x 20-bar average), and discrete position sizing (0.25). Enters long on bullish Donchian breakout above weekly R4 with volume confirmation; enters short on bearish breakdown below weekly S4 with volume confirmation. Exits on opposite Donchian breakout. Uses weekly HTF for structural bias to avoid counter-trend entries in ranging markets. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong momentum moves in both bull and bear regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = ~5 days of 6h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: weekly Camarilla pivot levels (R4, S4)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # S4 = C - Range * 1.1/2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # Align weekly levels to 6h timeframe (wait for weekly bar close)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Donchian channels (20-period) on 6h
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: bearish Donchian breakdown (price < Donchian low)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish Donchian breakout (price > Donchian high)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and weekly pivot alignment
            bullish_breakout = close[i] > donchian_high[i] and close[i] > r4_1w_aligned[i]
            bearish_breakout = close[i] < donchian_low[i] and close[i] < s4_1w_aligned[i]
            
            if bullish_breakout and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals