#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_New
Hypothesis: Donchian(20) breakout on 4h with 1d EMA34 trend filter, volume spike confirmation, and ATR trailing stop (2.5x). Uses discrete position sizing (0.25) to reduce fee drag. Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year). Works in both bull and bear markets by following 1d trend direction while using Donchian channels for precise breakout entries. ATR-based stoploss manages risk without look-ahead. Volume spike filter and increased ATR multiplier reduce trade frequency and improve Sharpe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # ATR for adaptive trailing stop (14-period ATR on 4h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 1d EMA34 (34) + Donchian (20) + volume avg (20) + ATR (14)
    start_idx = max(34, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        upper_ch = highest_20[i]
        lower_ch = lowest_20[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 1d EMA34 trend filter and volume spike
            # Long: price closes above upper Donchian AND above EMA34 (1d uptrend) AND volume spike
            long_condition = (close_val > upper_ch) and (close_val > ema_val) and vol_conf
            # Short: price closes below lower Donchian AND below EMA34 (1d downtrend) AND volume spike
            short_condition = (close_val < lower_ch) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches lower Donchian (opposite channel)
            # 2. 1d EMA34 turns bearish (price below EMA)
            # 3. ATR-based trailing stop: price drops 2.5 * ATR from highest since entry (increased from 2.0)
            exit_condition = (close_val < lower_ch) or (close_val < ema_val) or (close_val < highest_since_entry - 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches upper Donchian (opposite channel)
            # 2. 1d EMA34 turns bullish (price above EMA)
            # 3. ATR-based trailing stop: price rises 2.5 * ATR from lowest since entry (increased from 2.0)
            exit_condition = (close_val > upper_ch) or (close_val > ema_val) or (close_val > lowest_since_entry + 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_New"
timeframe = "4h"
leverage = 1.0