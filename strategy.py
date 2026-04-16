#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above Donchian upper band AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime).
# Short when price breaks below Donchian lower band AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime).
# Uses discrete position size 0.25. Donchian captures breakouts, volume confirms participation, chop filter ensures we only trade in ranging markets to avoid whipsaws in strong trends.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing meaningful moves.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for chop calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Choppiness Index(14) for regime filter ===
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_values = chop.values
    
    # Align 1d chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for Donchian, 20 for volume MA, 28 for chop)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        vol_spike = volume_spike[i]
        chop_val = chop_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to mid-band or volume spike ends
            mid_band = (upper_band + lower_band) / 2
            if price <= mid_band or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to mid-band or volume spike ends
            mid_band = (upper_band + lower_band) / 2
            if price >= mid_band or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > upper band AND volume spike AND chop > 61.8 (range regime)
            if price > upper_band and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < lower band AND volume spike AND chop > 61.8 (range regime)
            elif price < lower_band and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0