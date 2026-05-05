#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation
# Long when: price breaks above 4h Donchian(20) high AND 1d ATR(14)/ATR(50) > 1.2 (expanding volatility) AND volume > 1.8x 20-period MA
# Short when: price breaks below 4h Donchian(20) low AND 1d ATR(14)/ATR(50) > 1.2 AND volume > 1.8x 20-period MA
# Exit when: price returns to 4h Donchian(20) midpoint OR ATR regime contracts (ATR ratio < 0.8)
# Uses Donchian for structure, ATR regime for volatility expansion, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_1dATRRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 4h
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (highest_high + lowest_low) / 2.0
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = (close > highest_high) & (np.roll(close, 1) <= np.roll(highest_high, 1))
    donchian_breakout_down = (close < lowest_low) & (np.roll(close, 1) >= np.roll(lowest_low, 1))
    donchian_revert_mid = np.abs(close - donchian_mid) < 0.001 * close  # approximate midpoint return
    
    # Get 1d data ONCE before loop for ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for ATR(50)
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: expanding volatility when ATR(14) > 1.2 * ATR(50)
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0)
    atr_regime_expanding = atr_ratio > 1.2
    atr_regime_contracting = atr_ratio < 0.8  # exit condition
    
    # Align 1d ATR regime to 4h timeframe
    atr_regime_expanding_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_expanding.astype(float))
    atr_regime_contracting_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_contracting.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_regime_expanding_aligned[i]) or np.isnan(atr_regime_contracting_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + expanding ATR regime + volume filter
            if (donchian_breakout_up[i] and 
                atr_regime_expanding_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + expanding ATR regime + volume filter
            elif (donchian_breakout_down[i] and 
                  atr_regime_expanding_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR ATR regime contracts
            if (donchian_revert_mid[i] or atr_regime_contracting_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR ATR regime contracts
            if (donchian_revert_mid[i] or atr_regime_contracting_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals