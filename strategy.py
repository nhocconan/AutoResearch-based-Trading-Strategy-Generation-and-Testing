#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter
Hypothesis: 1h Camarilla pivot (R1/S1) breakouts filtered by 4h EMA34 trend and volume spike.
Enter long when price breaks above 1h R1 with 4h uptrend and above-average volume.
Enter short when price breaks below 1h S1 with 4h downtrend and above-average volume.
Exit on opposite level break or ATR(14) trailing stop (2.0*ATR).
Designed for low trade frequency (target: 15-37 trades/year on 1h) to minimize fee drag.
Uses 4h for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise.
Works in bull/bear via 4h trend alignment and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend, 1d for pivots)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1h Camarilla Pivot Levels (R1, S1) ===
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # Calculate pivots per 1h bar (using current 1h bar's HLC)
    camarilla_range = (high_1h - low_1h) * 1.1 / 12.0
    r1_1h = close_1h + camarilla_range
    s1_1h = close_1h - camarilla_range
    
    # === 4h EMA34 for HTF trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === Daily Camarilla Pivot Levels (R1, S1) for additional structure ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range_1d = (high_1d - low_1d) * 1.1 / 12.0
    r1_1d = close_1d + camarilla_range_1d
    s1_1d = close_1d - camarilla_range_1d
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Volume spike filter (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) 
            or np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i])):
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
        
        if position == 0:
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            
            # Long conditions: price > 1h R1, 4h uptrend, volume spike
            long_breakout = price > r1_1h[i]
            long_trend = price > ema_34_4h_aligned[i]
            
            # Short conditions: price < 1h S1, 4h downtrend, volume spike
            short_breakout = price < s1_1h[i]
            short_trend = price < ema_34_4h_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 1h S1 (support broken)
            elif price < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 1h R1 (resistance broken)
            elif price > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0