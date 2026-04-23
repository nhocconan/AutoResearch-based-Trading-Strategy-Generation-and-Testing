#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below lower Donchian channel AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price reverts to the midpoint of the Donchian channel or ATR-based stoploss hits.
Uses discrete position sizing (0.30) to minimize fee churn. Targets 15-25 trades/year per symbol.
Donchian channels provide clear breakout levels that work in both bull and bear markets.
1w EMA50 provides long-term trend filter that avoids counter-trend trades during major reversals.
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
    
    # Calculate Donchian channels on 1d data (using previous 20 periods)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_roll_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian channels to 1d timeframe (no additional delay needed as they're based on completed bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, high_roll_max)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, low_roll_min)
    donchian_midpoint_aligned = align_htf_to_ltf(prices, df_1d, (high_roll_max + low_roll_min) / 2.0)
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
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
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_midpoint_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND close > 1w EMA50 AND volume spike
            if (price > donchian_upper_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND close < 1w EMA50 AND volume spike
            elif (price < donchian_lower_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to midpoint or ATR stoploss
                if price <= donchian_midpoint_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_1d[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to midpoint or ATR stoploss
                if price >= donchian_midpoint_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_1d[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0