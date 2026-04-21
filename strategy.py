#!/usr/bin/env python3
"""
12h_1d_Donchian20_Breakout_Volume_ATRFilter_V1
Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR-based trend filter.
Works in bull/bear: In uptrend, buy breakouts above upper channel; in downtrend, sell breakdowns below lower channel.
ATR filter ensures we only trade when volatility is elevated (avoiding choppy markets). Volume confirms breakout strength.
Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for ATR and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[:-1] - close_1d[1:]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 1d EMA50 for trend filter (rising/falling)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = prices['high'].rolling(window=20, min_periods=20).max().values
    low_12h = prices['low'].rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # ATR filter: current ATR > 0.8 * 50-period average ATR (avoid low volatility chop)
        if i >= 50:
            atr_ma = np.nanmean(atr_14_aligned[i-50:i])
            atr_ok = not np.isnan(atr_ma) and atr_14_aligned[i] > 0.8 * atr_ma
        else:
            atr_ok = False
        
        # Trend filter: EMA50 direction
        ema_rising = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]) else False
        ema_falling = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]) else False
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume + ATR + rising EMA50 (uptrend)
            if (price > high_12h[i] and volume_ok and atr_ok and ema_rising):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume + ATR + falling EMA50 (downtrend)
            elif (price < low_12h[i] and volume_ok and atr_ok and ema_falling):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or EMA50 turns down
            if price < low_12h[i] or (i > 0 and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or EMA50 turns up
            if price > high_12h[i] or (i > 0 and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian20_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0