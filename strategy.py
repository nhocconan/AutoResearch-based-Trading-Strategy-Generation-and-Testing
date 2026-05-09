#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# Long when price breaks above Donchian(20) high, 1d ATR > 1d ATR(20) average, and volume > 2x average
# Short when price breaks below Donchian(20) low, 1d ATR > 1d ATR(20) average, and volume > 2x average
# Exit when price retraces to midpoint of Donchian channel
# Uses Donchian for trend-following breakouts, ATR filter to avoid low-volatility whipsaws, volume for conviction
# Designed to work in both bull and bear markets by filtering false breakouts during low volatility
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Donchian_Breakout_1dATR_Volatility_Filter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Donchian calculation (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # Calculate 20-period high and low
    donchian_high = prev_high.rolling(window=20, min_periods=20).max()
    donchian_low = prev_low.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high.values)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low.values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid.values)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate 1d ATR(20) average for comparison
    atr_ma_20 = atr_14.rolling(window=20, min_periods=20).mean()
    
    # Align ATR values to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14.values)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20.values)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_ma_20_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, ATR > ATR(20) average, volume spike
            if (close[i] > donchian_high_aligned[i] and 
                atr_14_aligned[i] > atr_ma_20_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, ATR > ATR(20) average, volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  atr_14_aligned[i] > atr_ma_20_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to midpoint of Donchian channel
            if close[i] <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to midpoint of Donchian channel
            if close[i] >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals