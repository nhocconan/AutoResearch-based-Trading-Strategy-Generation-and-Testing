#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d weekly ATR regime filter and volume confirmation
# Long when: price breaks above 6h BB(20,2) upper band AND 1d weekly ATR ratio > 1.2 (expanding volatility) AND volume > 1.5x 20-period MA
# Short when: price breaks below 6h BB(20,2) lower band AND 1d weekly ATR ratio > 1.2 AND volume > 1.5x 20-period MA
# Exit when: price returns to 6h BB(20,2) middle band OR opposite breakout occurs
# Uses Bollinger Bands for volatility-based breakouts, weekly ATR regime to filter choppy markets, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BollingerBreakout_1dATRRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 6h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Bollinger Bands on 6h (20,2)
    if len(close) >= 20:
        bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
        bb_upper = bb_middle + 2.0 * bb_std
        bb_lower = bb_middle - 2.0 * bb_std
    else:
        bb_middle = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
    
    # Bollinger Band breakout signals
    bb_breakout_up = (close > bb_upper) & (np.roll(close, 1) <= np.roll(bb_upper, 1))
    bb_breakout_down = (close < bb_lower) & (np.roll(close, 1) >= np.roll(bb_lower, 1))
    bb_revert_middle = np.abs(close - bb_middle) < 0.001 * close  # approximate middle band return
    
    # Get 1d data ONCE before loop for weekly ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for ATR calculation
        return np.zeros(n)
    
    # Calculate ATR on 1d (14-period)
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        tr1 = np.abs(high_1d - low_1d)
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr1[0] = 0  # first value has no previous close
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        # Calculate ATR ratio: current ATR / 50-period MA of ATR (regime filter)
        if len(atr_14) >= 50:
            atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
            atr_ratio = atr_14 / atr_ma_50
            # Expanding volatility regime: ATR ratio > 1.2
            volatility_expanding = atr_ratio > 1.2
        else:
            volatility_expanding = np.full(len(df_1d), False)
    else:
        volatility_expanding = np.full(len(df_1d), False)
    
    # Align 1d volatility regime to 6h timeframe
    volatility_expanding_aligned = align_htf_to_ltf(prices, df_1d, volatility_expanding.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(volatility_expanding_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: BB breakout up + volatility expanding + volume filter
            if (bb_breakout_up[i] and 
                volatility_expanding_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: BB breakout down + volatility expanding + volume filter
            elif (bb_breakout_down[i] and 
                  volatility_expanding_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to BB middle OR short breakout occurs
            if (bb_revert_middle[i] or bb_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to BB middle OR long breakout occurs
            if (bb_revert_middle[i] or bb_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals