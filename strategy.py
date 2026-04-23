#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
Long when price breaks above 6h Donchian upper band AND weekly pivot is bullish (close > weekly PP) AND volume > 1.5x 20-period average.
Short when price breaks below 6h Donchian lower band AND weekly pivot is bearish (close < weekly PP) AND volume > 1.5x 20-period average.
Exit when price retraces to 6h Donchian midpoint OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~25 trades/year on 6h timeframe.
Weekly pivot provides higher timeframe bias to avoid counter-trend trades in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly HTF data for pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot point (PP) from prior week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla-inspired pivot (standard PP)
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(lookback, 20, 14)  # Donchian needs 20, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        pp_val = pp_1w_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND weekly PP bullish (close > weekly PP) AND volume spike
            if close[i] > upper and close[i] > pp_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower AND weekly PP bearish (close < weekly PP) AND volume spike
            elif close[i] < lower and close[i] < pp_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: Price retraces to Donchian midpoint
            if position == 1 and close[i] <= mid:
                exit_signal = True
            elif position == -1 and close[i] >= mid:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation_MidExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0