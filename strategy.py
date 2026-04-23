#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and ATR-based stoploss.
Long when price breaks above upper Donchian channel and close > 1d EMA34 (uptrend) with volume > 1.5x average.
Short when price breaks below lower Donchian channel and close < 1d EMA34 (downtrend) with volume > 1.5x average.
Exit on opposite Donchian breakout or ATR trailing stop. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Donchian channels provide clear structure, EMA34 filters trend direction, volume confirms breakout conviction.
ATR stoploss manages risk. Works in both bull and bear markets by aligning with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(arr, period):
    """Exponential Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_vals = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_vals

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = ema(close_1d, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) for stoploss on primary timeframe
    atr_14 = atr(high, low, close, 14)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        atr_val = atr_14[i]
        vol_ma_val = vol_ma_20[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1d EMA34 (uptrend) AND volume confirmation
            if (price > upper_donchian and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below lower Donchian AND price < 1d EMA34 (downtrend) AND volume confirmation
            elif (price < lower_donchian and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR ATR trailing stop (2.5 * ATR below highest)
                if price < lower_donchian or price < (highest_since_entry - 2.5 * atr_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR ATR trailing stop (2.5 * ATR above lowest)
                if price > upper_donchian or price > (lowest_since_entry + 2.5 * atr_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0