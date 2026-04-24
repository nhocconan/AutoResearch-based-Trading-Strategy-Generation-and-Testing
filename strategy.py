#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
- Primary timeframe: 4h for execution.
- HTF: 1d for ATR regime filter (high volatility = trend following, low volatility = avoid).
- Donchian channels calculated from 20-period high/low on 4h.
- Entry: Long when price breaks above upper Donchian with volume spike and 1d ATR > 20-period MA of ATR.
         Short when price breaks below lower Donchian with volume spike and 1d ATR > 20-period MA of ATR.
- Exit: When price returns to the middle of the Donchian channel (mean reversion).
- Uses discrete signal size 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d True Range and ATR(20)
    tr1 = pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['close'])
    tr3 = pd.Series(df_1d['close']).shift(1) - pd.Series(df_1d['low'])
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period MA of 1d ATR for regime filter
    atr_ma_20 = pd.Series(atr_20).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR indicators to 4h
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # Donchian channels on 4h: 20-period high/low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_20_aligned[i]) or 
            np.isnan(atr_ma_20_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and volatility filter
            if volume_spike[i] and atr_20_aligned[i] > atr_ma_20_aligned[i]:
                # Bullish breakout: price > upper Donchian
                if close[i] > high_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < lower Donchian
                elif close[i] < low_20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to Donchian middle (mean reversion)
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian middle (mean reversion)
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRVolFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0