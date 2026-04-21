#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeFilter_v2
Hypothesis: 6h Donchian(20) breakout filtered by 1w pivot direction (R1/S1) and volume spike.
In weekly uptrend (price > weekly R1): long breakouts above 20-period high.
In weekly downtrend (price < weekly S1): short breakouts below 20-period low.
Volume confirmation (>1.5x average) filters false breakouts. Works in both bull/bear by aligning with weekly structure.
Discrete position sizing (0.25) and ATR(14) stoploss (2.0x) limit fee drag and drawdown.
Target: 75-150 total trades over 4 years = 19-38/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for weekly pivot via Camarilla on 1d resampled? No, use actual 1w from mtf_data)
    # Since we need weekly pivot, get 1d data then compute weekly Camarilla from it? But mtf_data only gives specific TFs.
    # Alternative: use 1d to approximate weekly? Not accurate.
    # Check available TFs: 5m,15m,30m,1h,4h,6h,12h,1d, HTF ref: 1w
    # So we can get 1w data via get_htf_data(prices, '1w')? The description says HTF ref: all above + 1w.
    # Yes, 1w is available as HTF reference.
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w OHLC for Camarilla pivot (R1/S1) ===
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    range_1w = df_1w_high - df_1w_low
    r1_1w = df_1w_close + 0.275 * range_1w
    s1_1w = df_1w_close - 0.275 * range_1w
    
    # Align 1w Camarilla levels to 6h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 6h Donchian(20) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: 20-period high/low
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) 
            or np.isnan(dc_high[i]) or np.isnan(dc_low[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1w = r1_1w_aligned[i]
        s1w = s1_1w_aligned[i]
        dch = dc_high[i]
        dcl = dc_low[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Weekly uptrend: price > weekly R1 -> look for long breakouts
            # Weekly downtrend: price < weekly S1 -> look for short breakouts
            weekly_uptrend = price > r1w
            weekly_downtrend = price < s1w
            
            long_condition = weekly_uptrend and (price > dch) and volume_confirmed
            short_condition = weekly_downtrend and (price < dcl) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss: 2.0x ATR
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if weekly trend reverses
            elif price < r1w:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at Donchian low (oversold bounce)
            elif price < dcl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss: 2.0x ATR
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if weekly trend reverses
            elif price > s1w:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at Donchian high (overbought bounce)
            elif price > dch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeFilter_v2"
timeframe = "6h"
leverage = 1.0