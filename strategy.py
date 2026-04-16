#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1d ATR regime filter.
# Long when price breaks above Donchian upper band AND 1d volume > 1.5x 20-period average AND 1d ATR(10) > ATR(30) (expanding volatility).
# Short when price breaks below Donchian lower band AND 1d volume > 1.5x 20-period average AND 1d ATR(10) > ATR(30).
# Exit when price returns to Donchian midpoint or ATR(10) < ATR(30) (contracting volatility).
# Uses discrete position size 0.25. Donchian provides structure, volume confirmation reduces false signals,
# and ATR regime ensures we only trade in volatile, breakout-prone markets. Target: 50-120 total trades over 4 years (12-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for volume and ATR filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume MA and ATR regime ===
    # 20-period volume average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Align 1d indicators to 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    atr_30_aligned = align_htf_to_ltf(prices, df_1d, atr_30)
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Donchian channels (20-period) ===
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 12h timeframe (no additional shift needed)
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_12h, middle_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(middle_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_10_aligned[i]) or np.isnan(atr_30_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        middle_val = middle_20_aligned[i]
        vol_ma_val = vol_ma_20_aligned[i]
        atr_10_val = atr_10_aligned[i]
        atr_30_val = atr_30_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average (using 1d volume MA aligned to 12h)
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # ATR regime filter: ATR(10) > ATR(30) (expanding volatility)
        atr_filter = atr_10_val > atr_30_val
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to midpoint or ATR contracts
            if price <= middle_val or atr_10_val <= atr_30_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to midpoint or ATR contracts
            if price >= middle_val or atr_10_val <= atr_30_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper band with volume and ATR regime confirmation
            if price > upper_val and vol_filter and atr_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower band with volume and ATR regime confirmation
            elif price < lower_val and vol_filter and atr_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_1dATRRegime_V1"
timeframe = "12h"
leverage = 1.0