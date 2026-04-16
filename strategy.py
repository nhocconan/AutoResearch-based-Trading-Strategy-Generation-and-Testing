#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h choppiness regime filter.
# Long when price breaks above Donchian upper (20) AND 12h volume > 1.5x 20-period average AND 12h chop > 61.8 (range regime).
# Short when price breaks below Donchian lower (20) AND 12h volume > 1.5x 20-period average AND 12h chop > 61.8 (range regime).
# Uses discrete position size 0.25. Donchian captures breakouts in ranging markets, volume confirms participation, chop filter ensures we only trade in ranging conditions (avoiding trends where breakouts fail).
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets during ranging conditions.
# Target: 75-150 trades over 4 years (19-38/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for chop calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Choppiness Index (14) ===
    # True Range
    tr1 = pd.Series(high_12h).diff()
    tr2 = pd.Series(low_12h).diff().abs()
    tr3 = pd.Series(close_12h).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_values = chop.values
    
    # Align 12h indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for Donchian/vol MA/chop)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Donchian midpoint or volume spike ends
            midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
            if price <= midpoint or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Donchian midpoint or volume spike ends
            midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
            if price >= midpoint or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Donchian upper AND volume spike AND chop > 61.8 (range regime)
            if price > donchian_upper[i] and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < Donchian lower AND volume spike AND chop > 61.8 (range regime)
            elif price < donchian_lower[i] and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0