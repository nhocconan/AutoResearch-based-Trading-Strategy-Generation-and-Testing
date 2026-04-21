#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Trend_Volume_V1
Hypothesis: Breakout above Donchian(20) high in 12h timeframe with 1w trend filter (EMA50) and volume confirmation works in both bull and bear markets. Uses 1w EMA for trend filter to avoid counter-trend trades. Volume spike confirms breakout strength. ATR-based stoploss via signal=0 when price closes below 2*ATR from entry.
Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w ATR(14) for stoploss calculation
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate Donchian(20) on 12h data (primary timeframe)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian(20) upper band: highest high of last 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian(20) lower band: lowest low of last 20 periods
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close_12h[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND 1w uptrend AND volume spike
            if (price > donchian_high[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and  # 1w EMA rising
                volume_ok):
                signals[i] = 0.25
                position = 1
                entry_price = price
        
        elif position == 1:
            # Long exit: stoploss (price closes below entry - 2*ATR) OR trend reversal (1w EMA falling)
            stop_price = entry_price - 2.0 * atr_14_1w_aligned[i]
            if price < stop_price or ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
    
    return signals

name = "12h_Donchian20_Breakout_Trend_Volume_V1"
timeframe = "12h"
leverage = 1.0