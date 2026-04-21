#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1
Hypothesis: 4h Donchian channel (20-period) breakout filtered by 12h EMA50 trend and volume spike (2.0x average).
Long when price breaks above Donchian upper band and above 12h EMA50; short when price breaks below Donchian lower band and below 12h EMA50.
Volume confirmation reduces false breakouts. ATR(14) stoploss (2.0x) and discrete sizing (0.25).
Designed to work in both bull and bear markets via 12h trend alignment and strict entry filters.
Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 4h Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper: highest high of last 20 periods
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low of last 20 periods
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (50-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) 
            or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Only enter in trending markets (price > 12h EMA50 for long, < for short)
            # Volume confirmation required to avoid false breakouts
            long_condition = (price > upper) and (price > ema_trend) and volume_confirmed
            short_condition = (price < lower) and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at Donchian upper (extreme overbought)
            elif price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at Donchian lower (extreme oversold)
            elif price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0