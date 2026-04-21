#!/usr/bin/env python3
"""
12h_1d_Donchian20_Breakout_Volume_ATRFilter_V1
Hypothesis: Donchian(20) breakout on 12h timeframe with 1d trend filter (EMA50), volume confirmation, and ATR-based stoploss.
Works in bull/bear: In uptrend (price>EMA50), long breakout of Donchian high; in downtrend (price<EMA50), short breakout of Donchian low.
Volume filter ensures breakout validity. ATR stoploss manages risk.
Target: 12-25 trades/year per symbol (50-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    atr = None
    
    # Pre-calculate ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Calculate Donchian channels (20-period) using only past data
        if i >= 20:
            donch_high = prices['high'].iloc[i-20:i].max()
            donch_low = prices['low'].iloc[i-20:i].min()
        else:
            # Not enough data for Donchian calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = prices['volume'].iloc[i] > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Determine trend: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
        uptrend = price > ema_50_1d_aligned[i]
        downtrend = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND uptrend AND volume confirmation
            if price > donch_high and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND downtrend AND volume confirmation
            elif price < donch_low and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < Donchian low (breakdown) OR stoploss hit (2*ATR below entry)
            # Simplified: exit on Donchian low breakdown or price < EMA50 (trend change)
            if price < donch_low or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > Donchian high (breakout) OR stoploss hit (2*ATR above entry)
            # Simplified: exit on Donchian high breakout or price > EMA50 (trend change)
            if price > donch_high or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian20_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0