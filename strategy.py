#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation (>2.0x 20-period average).
# Long when price breaks above Donchian upper AND 1d ATR(14) > 1.5x its 50-period MA (high volatility regime) AND volume > 2.0x MA20.
# Short when price breaks below Donchian lower AND 1d ATR(14) > 1.5x its 50-period MA AND volume > 2.0x MA20.
# Exit when price crosses the Donchian midline (average of upper and lower).
# Uses 1d HTF for volatility regime filter to avoid low-volatility chop and reduce false breakouts.
# Higher volume threshold (2.0x) and volatility filter target 75-150 total trades over 4 years for 4h timeframe.
# Donchian breakouts capture strong momentum moves; volatility filter ensures trading only during active markets.

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Donchian Channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: > 2.0x 20-period average (high threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) - volatility regime filter
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[:-1] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    high_volatility = atr_14_1d > (1.5 * atr_ma_50_1d)
    
    # Align HTF indicators to LTF
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    high_volatility_aligned = align_htf_to_ltf(prices, df_1d, high_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(high_volatility_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper AND high volatility regime AND volume confirm
            if (close[i] > donchian_upper[i] and 
                high_volatility_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower AND high volatility regime AND volume confirm
            elif (close[i] < donchian_lower[i] and 
                  high_volatility_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian middle (trend weakening)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian middle (trend weakening)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals