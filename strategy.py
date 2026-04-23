#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume spike.
Long when price breaks above 20-day Donchian high AND close > 1w EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below 20-day Donchian low AND close < 1w EMA34 AND volume > 2.0x 20-period average.
Exit when price reverts to 10-day Donchian midpoint or ATR-based stoploss (2.0x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 10-25 trades/year per symbol.
Donchian channels provide robust trend-following structure, while 1w EMA34 filters for higher-timeframe trend alignment.
Volume spike confirms institutional participation. ATR stoploss manages risk. Works in both bull and bear markets by avoiding counter-trend breakouts.
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
    
    # Load 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 1d Donchian levels to 1d timeframe (use previous day's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 1d data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND close > 1w EMA34 AND volume spike
            if (price > donchian_high_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low AND close < 1w EMA34 AND volume spike
            elif (price < donchian_low_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to midpoint or ATR stoploss
                if price <= donchian_mid_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_1d[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to midpoint or ATR stoploss
                if price >= donchian_mid_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_1d[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0