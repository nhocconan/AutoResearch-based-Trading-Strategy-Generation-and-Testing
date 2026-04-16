#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1d ATR regime filter.
# Long when price breaks above Donchian upper AND volume > 1.5x 20-period average AND ATR(10) > ATR(30) (expanding volatility).
# Short when price breaks below Donchian lower AND volume > 1.5x 20-period average AND ATR(10) > ATR(30).
# Exit when ATR(10) < ATR(30) (contracting volatility) or opposite Donchian break.
# Uses discrete position size 0.25. Donchian provides structure, volume confirms participation, ATR regime ensures trending markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for ATR regime and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: ATR(10) and ATR(30) for regime filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) and ATR(30)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR regime: expanding volatility (trending market)
    atr_regime = atr_10 > atr_30
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=1).mean().values
    vol_filter = volume_1d > 1.5 * vol_ma_20
    
    # Align 1d indicators to 12h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Get 12h data for Donchian channels (using primary timeframe data directly)
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_regime_aligned[i]) or np.isnan(vol_filter_aligned[i]) or 
            np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        atr_regime_val = atr_regime_aligned[i]
        vol_filter_val = vol_filter_aligned[i]
        upper = high_roll_max[i]
        lower = low_roll_min[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if ATR regime contracts (ending trend) or price breaks lower Donchian
            if not atr_regime_val or price < lower:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if ATR regime contracts (ending trend) or price breaks upper Donchian
            if not atr_regime_val or price > upper:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper with volume and ATR regime confirmation
            if price > upper and vol_filter_val and atr_regime_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower with volume and ATR regime confirmation
            elif price < lower and vol_filter_val and atr_regime_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolSpike_ATRRegime_V1"
timeframe = "12h"
leverage = 1.0