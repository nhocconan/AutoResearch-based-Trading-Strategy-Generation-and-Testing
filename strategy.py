#!/usr/bin/env python3
"""
4h_Donchian20_Trend_EMA50_VolumeSpike_ATRStop_v1
Hypothesis: 4h Donchian(20) breakout filtered by 1d EMA50 trend and volume spike (>2.0x 20-period average).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to balance returns and fee drag.
Designed to work in both bull and bear markets via 1d trend filter and volatility-adjusted exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    df_1d_close = df_1d['close'].values
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Donchian(20) on 4h timeframe ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(prices['close'].values, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(prices['close'].values, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter: 20-period average (moderate) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) 
            or np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        if position == 0:
            # Volume filter: current volume > 2.0x 20-period average
            vol_filter = vol_current > 2.0 * vol_average
            
            # Long conditions: price > Donchian high (breakout), 1d uptrend, volume filter
            long_breakout = price > donchian_high[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < Donchian low (breakdown), 1d downtrend, volume filter
            short_breakout = price < donchian_low[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic - balanced filters for moderate trade frequency
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Donchian low
            elif price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Donchian high
            elif price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Trend_EMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0