#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR regime filter (trending when ATR7/ATR30 > 0.8) + volume spike confirmation
# Long when: price breaks above 4h Donchian(20) high AND 1d ATR regime indicates trending (ATR7/ATR30 > 0.8) AND volume > 1.5x 20-period MA
# Short when: price breaks below 4h Donchian(20) low AND 1d ATR regime indicates trending (ATR7/ATR30 > 0.8) AND volume > 1.5x 20-period MA
# Exit when: price returns to 4h Donchian(20) midpoint OR opposite breakout occurs
# Uses Donchian for structure, ATR regime for trend detection (works in bull/bear), volume for conviction
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
    
    # Get 1d data ONCE before loop for ATR regime calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for ATR30
        return np.zeros(n)
    
    # Calculate ATR on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Handle first bar
    tr[0] = tr1[0]
    
    # ATR(7) and ATR(30)
    if len(tr) >= 30:
        atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
        atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
        # Avoid division by zero
        atr_ratio = np.where(atr_30 != 0, atr_7 / atr_30, 0)
        # Trending regime: ATR7/ATR30 > 0.8 indicates strong trending conditions
        atr_regime_trending = atr_ratio > 0.8
    else:
        atr_ratio = np.zeros(len(df_1d))
        atr_regime_trending = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d ATR regime to 4h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_trending.astype(float))
    
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
            # Long conditions: Donchian breakout up + ATR regime trending + volume filter
            if (donchian_breakout_up[i] and 
                atr_regime_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + ATR regime trending + volume filter
            elif (donchian_breakout_down[i] and 
                  atr_regime_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR short breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR long breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals