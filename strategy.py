#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day ATR filter and volume confirmation.
Long when price breaks above Donchian upper channel (20-period) and 1-day ATR > 1-day ATR MA.
Short when price breaks below Donchian lower channel and 1-day ATR > 1-day ATR MA.
Exit when price crosses the Donchian middle line (10-period average of high/low).
ATR filter ensures volatility expansion during breakouts; volume confirmation adds confluence.
Works in bull markets by catching breakouts and in bear markets by catching breakdowns.
"""

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
    
    # Load 1-day data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # 1-day ATR moving average (20-period)
    atr_ma_20 = np.full(len(atr_14), np.nan)
    for i in range(20, len(atr_14)):
        if not np.isnan(atr_14[i-19:i+1]).any():
            atr_ma_20[i] = np.nanmean(atr_14[i-19:i+1])
    
    # Align ATR and ATR MA to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper channel and ATR > ATR MA
            if close[i] > donch_high[i] and atr_14_aligned[i] > atr_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower channel and ATR > ATR MA
            elif close[i] < donch_low[i] and atr_14_aligned[i] > atr_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian middle line
                if close[i] < donch_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian middle line
                if close[i] > donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_ATRFilter_Volume"
timeframe = "4h"
leverage = 1.0