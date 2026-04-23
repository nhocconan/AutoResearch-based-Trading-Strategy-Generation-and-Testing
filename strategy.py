#!/usr/bin/env python3
"""
Hypothesis: 1h session-filtered 4h Donchian breakout with 1d EMA200 trend filter and volume spike confirmation.
Long when price breaks above 4h upper Donchian AND close > 1d EMA200 AND volume > 2.5x 20-period average AND hour in 08-20 UTC.
Short when price breaks below 4h lower Donchian AND close < 1d EMA200 AND volume > 2.5x 20-period average AND hour in 08-20 UTC.
Exit when price reverts to 4h middle Donchian band (20-period mean) or ATR-based stoploss hits.
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-37 trades/year per symbol.
1h timeframe with session filter reduces noise; 4h/1d HTF provides robust trend/structure for both bull and bear markets.
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
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels for 4h timeframe (using previous bar's data)
    lookback = 20
    upper_4h = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().values
    lower_4h = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().values
    middle_4h = pd.Series(close_4h).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Align 4h Donchian levels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Load 1d data for EMA200 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 1h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 1h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(middle_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper DONCHIAN AND close > 1d EMA200 AND volume spike AND in session
            if (price > upper_4h_aligned[i] and 
                close[i] > ema200_1d_aligned[i] and 
                volume[i] > 2.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below lower DONCHIAN AND close < 1d EMA200 AND volume spike AND in session
            elif (price < lower_4h_aligned[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume[i] > 2.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle band or ATR stoploss
                if price <= middle_4h_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle band or ATR stoploss
                if price >= middle_4h_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Session_Donchian20_1dEMA200_VolumeSpike"
timeframe = "1h"
leverage = 1.0