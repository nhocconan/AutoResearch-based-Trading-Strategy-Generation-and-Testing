#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above Donchian upper AND 1d ATR ratio (ATR7/ATR30) < 0.8 (low vol) AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower AND 1d ATR ratio < 0.8 AND volume > 1.5x 20-period average.
Exit when price retouches Donchian midpoint or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag. Targets 15-30 trades/year per symbol.
Donchian provides clear trend structure; low volatility filter avoids choppy markets; volume confirms breakout conviction.
Works in bull (breakouts with volume) and bear (breakdowns with volume) markets by filtering for low volatility regimes where breakouts are more reliable.
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
    
    # Calculate 1d ATR(7) and ATR(30) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # first bar
    
    atr7_1d = pd.Series(tr_1d).rolling(window=7, min_periods=7).mean().values
    atr30_1d = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio_1d = atr7_1d / atr30_1d  # < 0.8 = low volatility regime
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Donchian(20) channels on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation on 4h
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 20)  # ATR30 needs 30, Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        atr_ratio_val = atr_ratio_1d_aligned[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        mid = donchian_mid[i]
        
        if position == 0:
            # Long: Break above upper AND low volatility regime AND volume confirmation
            if close[i] > upper and atr_ratio_val < 0.8 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below lower AND low volatility regime AND volume confirmation
            elif close[i] < lower and atr_ratio_val < 0.8 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Donchian midpoint
            if position == 1 and close[i] <= mid:
                exit_signal = True
            elif position == -1 and close[i] >= mid:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_Ratio_LowVol_VolumeConfirmation_MidpointExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0