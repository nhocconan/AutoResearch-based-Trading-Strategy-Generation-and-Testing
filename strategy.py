#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper (20) AND price > 1d EMA34 (uptrend) AND volume > 1.5x average.
Short when price breaks below Donchian lower (20) AND price < 1d EMA34 (downtrend) AND volume > 1.5x average.
Exit when price reverts to Donchian midpoint or trend reverses (price crosses 1d EMA34).
Uses ATR-based stoploss to limit drawdown. Designed for ~20-40 trades/year to avoid fee drag while capturing strong breakouts.
Works in both bull and bear markets by requiring 1d EMA34 trend confirmation for entries.
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
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 for 1d trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_ma + low_ma) / 2
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]]) if len(tr1) > 0 else [0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        ema34_val = ema34_aligned[i]
        upper = high_ma[i]
        lower = low_ma[i]
        midpoint = donchian_mid[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > upper and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < lower and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to midpoint OR price breaks below 1d EMA34 (trend reversal) OR stoploss hit
                if price <= midpoint or price < ema34_val or price <= entry_price - 2.0 * atr_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to midpoint OR price breaks above 1d EMA34 (trend reversal) OR stoploss hit
                if price >= midpoint or price > ema34_val or price >= entry_price + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0