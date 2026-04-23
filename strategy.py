#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
- Donchian(20) from 1d provides robust structure for breakouts in both bull and bear markets
- 1d ATR(14) filters low-volatility chop: only trade when ATR > its 50-period MA (expanding volatility)
- Volume confirmation (> 1.5x 24-period MA) reduces false breakouts
- Designed for 12h timeframe to capture medium-term moves with controlled frequency (target: 12-37 trades/year)
- Works in bull via long breakouts above upper band and in bear via short breakdowns below lower band
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels: upper = max(high,20), lower = min(low,20)
    donch_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (use previous day's channels for breakout)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR and its MA to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Volume confirmation: > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 24)  # need Donchian(20), ATR MA(50), vol MA(24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND expanding volatility AND volume spike
            if (close[i] > donch_upper_aligned[i] and 
                atr_14_aligned[i] > atr_ma_50_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND expanding volatility AND volume spike
            elif (close[i] < donch_lower_aligned[i] and 
                  atr_14_aligned[i] > atr_ma_50_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band OR volatility contracts
            exit_signal = False
            if position == 1:
                # Exit long when price < lower Donchian OR volatility contracts
                if close[i] < donch_lower_aligned[i] or atr_14_aligned[i] <= atr_ma_50_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > upper Donchian OR volatility contracts
                if close[i] > donch_upper_aligned[i] or atr_14_aligned[i] <= atr_ma_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "12h"
leverage = 1.0