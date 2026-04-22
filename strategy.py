#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR-based volatility filter and volume confirmation.
# Uses Donchian(20) channels for breakout detection, filtered by 1d ATR volatility regime
# (low volatility = breakout more likely to trend) and volume spike (>1.5x 20-period average).
# Designed for low trade frequency (~20-40/year) to minimize fee decay.
# Works in both bull and bear markets by requiring volatility regime filter and volume confirmation.
# Entry: Price breaks above/below Donchian channel + low volatility regime + volume spike.
# Exit: Price returns to middle of Donchian channel or volatility regime shifts to high.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ATR calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d for volatility regime
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d ATR to 4h timeframe (waits for 1d bar to close)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr_14_aligned[i]
        
        # Volatility regime: low volatility = ATR below its 50-period median (calculated on 1d)
        # For simplicity, use ATR < 50-period SMA of ATR as low vol regime
        if i >= 50:
            atr_ma_50 = np.nanmedian(atr_14_aligned[max(0, i-49):i+1])  # rolling median of ATR
            low_vol_regime = atr_val < atr_ma_50
        else:
            low_vol_regime = False
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + low vol regime + volume spike
            if price > donchian_high[i] and low_vol_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + low vol regime + volume spike
            elif price < donchian_low[i] and low_vol_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to middle of Donchian channel or volatility shifts to high
                if price < donchian_mid[i] or not low_vol_regime:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to middle of Donchian channel or volatility shifts to high
                if price > donchian_mid[i] or not low_vol_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Volume"
timeframe = "4h"
leverage = 1.0