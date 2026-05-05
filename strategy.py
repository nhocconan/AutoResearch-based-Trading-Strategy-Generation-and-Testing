#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when: price breaks above 4h Donchian(20) high AND 1d ATR(14)/ATR(50) < 0.8 (low vol regime) AND volume > 1.5x 20-period MA
# Short when: price breaks below 4h Donchian(20) low AND 1d ATR(14)/ATR(50) < 0.8 AND volume > 1.5x 20-period MA
# Exit when: price returns to 4h Donchian(20) midpoint
# Uses Donchian for structure, ATR regime to avoid whipsaws in high volatility, volume for conviction
# Timeframe: 4h. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

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
        volume_filter = volume > (1.5 * vol_ma_20)
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
    
    # Calculate 1d ATR(14) and ATR(50)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    if len(tr) >= 50:
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
        # Avoid division by zero
        atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 1.0)
        atr_regime_filter = atr_ratio < 0.8  # low volatility regime
    else:
        atr_regime_filter = np.full(len(df_1d), False)
    
    # Align 1d ATR regime to 4h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_regime_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + ATR regime + volume filter
            if (donchian_breakout_up[i] and 
                atr_regime_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + ATR regime + volume filter
            elif (donchian_breakout_down[i] and 
                  atr_regime_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint
            if donchian_revert_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint
            if donchian_revert_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals