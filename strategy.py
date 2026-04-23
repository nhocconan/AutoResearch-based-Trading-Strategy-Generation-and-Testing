#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR(14) filter for trend strength.
Long when price breaks above Donchian upper AND volume > 1.5x 20-period MA AND ATR > ATR MA.
Short when price breaks below Donchian lower AND volume > 1.5x 20-period MA AND ATR > ATR MA.
Exit when price touches Donchian middle (mean of upper/lower) or ATR drops below filter.
Uses 1d HTF for trend filter (price > 1d EMA50 for long, price < 1d EMA50 for short) to avoid counter-trend trades.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) and its MA(10) for trend strength filter
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 4h volume > 1.5x 20-period MA (tight threshold to reduce trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # ATR filter: current ATR > its MA (ensures sufficient volatility/trend strength)
        atr_filter = atr[i] > atr_ma[i]
        
        if position == 0:
            # Long: price > upper AND 1d EMA50 uptrend (price > EMA50) AND volume filter AND ATR filter
            if close[i] > donchian_upper[i] and close[i] > ema_50_1d_aligned[i] and vol_filter and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: price < lower AND 1d EMA50 downtrend (price < EMA50) AND volume filter AND ATR filter
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1d_aligned[i] and vol_filter and atr_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches middle OR 1d EMA50 trend turns down (price < EMA50)
                if close[i] <= donchian_middle[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches middle OR 1d EMA50 trend turns up (price > EMA50)
                if close[i] >= donchian_middle[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA50_Trend_VolumeATR_Filter"
timeframe = "4h"
leverage = 1.0