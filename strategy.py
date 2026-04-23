#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below lower Donchian AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when price reverts to midpoint of Donchian channel or ATR-based stoploss.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Donchian channels provide clear structure, 1d EMA50 filters for long-term trend, volume confirms breakout strength.
Designed to work in both bull (breakouts continuation) and bear (breakouts reversal) markets via trend filter.
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
    
    # Load 12h data for Donchian calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) on 12h data (using previous 20 completed bars)
    # Upper = max(high of past 20 bars), Lower = min(low of past 20 bars)
    high_roll_max = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, high_roll_max)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, low_roll_min)
    donchian_mid = (high_roll_max + low_roll_min) / 2.0
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 12h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND close > 1d EMA50 AND volume spike
            if (price > donchian_upper_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND close < 1d EMA50 AND volume spike
            elif (price < donchian_lower_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
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
                elif price < entry_price - 2.5 * atr_12h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to midpoint or ATR stoploss
                if price >= donchian_mid_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA50_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0