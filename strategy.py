#!/usr/bin/env python3
"""
1h_HTF_4h_1d_Camarilla_R1S1_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use 1h primary timeframe with 4h trend filter (EMA34) and 1d Camarilla R1/S1 breakout for signal direction.
Volume confirmation (>1.8x 20-bar volume MA) reduces false breakouts. ATR trailing stop (2.0x ATR) limits drawdown.
Position size 0.20 keeps risk controlled. Target 15-37 trades/year per symbol to avoid fee drag.
Uses 4h EMA34 for trend bias (long only above EMA34, short only below) to work in bull/bear via directional filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 34 or len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = pivot + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pivot - (high_1d - low_1d) * 1.1 / 12.0
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is datetime64[ms], index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0  # for long trailing stop
    lowest_low_since_entry = 0.0    # for short trailing stop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.8 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above 1d R1 + volume confirmation + 4h EMA34 uptrend
            if price > r1_aligned[i-1] and vol_ok and price > ema_34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below 1d S1 + volume confirmation + 4h EMA34 downtrend
            elif price < s1_aligned[i-1] and vol_ok and price < ema_34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            # ATR trailing stop: exit if price drops 2.0*ATR from highest high since entry
            if price < highest_high_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Update lowest low since entry
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest low since entry
            if price > lowest_low_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_HTF_4h_1d_Camarilla_R1S1_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "1h"
leverage = 1.0