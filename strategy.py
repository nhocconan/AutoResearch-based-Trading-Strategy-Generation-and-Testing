#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above R3 AND close > 1d EMA34 AND volume > 1.8x 20-period average.
Short when price breaks below S3 AND close < 1d EMA34 AND volume > 1.8x 20-period average.
Exit when price reverts to H3/L3 level or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-35 trades/year per symbol.
12h timeframe reduces trade frequency vs 4h, while 1d EMA34 provides robust higher-timeframe trend filter
that works in both bull and bear markets by avoiding counter-trend breakouts.
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
    
    # Load 12h data for Camarilla calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for 12h timeframe (using previous bar's data)
    lookback = 20
    hh_12h = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    ll_12h = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    close_12h = df_12h['close'].values
    range_12h = hh_12h - ll_12h
    
    # Camarilla levels: H3, L3, H4, L4
    # H3 = close + range * 1.1/4
    # L3 = close - range * 1.1/4
    # H4 = close + range * 1.1/2
    # L4 = close - range * 1.1/2
    h3_12h = close_12h + range_12h * 1.1 / 4
    l3_12h = close_12h - range_12h * 1.1 / 4
    h4_12h = close_12h + range_12h * 1.1 / 2
    l4_12h = close_12h - range_12h * 1.1 / 2
    
    # Align 12h Camarilla levels to 12h timeframe
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
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
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or 
            np.isnan(h4_12h_aligned[i]) or np.isnan(l4_12h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above H3 AND close > 1d EMA34 AND volume spike
            if (price > h3_12h_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below L3 AND close < 1d EMA34 AND volume spike
            elif (price < l3_12h_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to L3 or ATR stoploss
                if price <= l3_12h_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_12h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to H3 or ATR stoploss
                if price >= h3_12h_aligned[i]:
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

name = "12H_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0