#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: 6h Camarilla pivot (R1/S1) breakout filtered by weekly EMA20 trend and volume spike (>2.0x 20-period average).
Weekly trend provides robust multi-month directional bias, reducing whipsaws in both bull and bear markets.
Volume confirmation ensures breakout validity. ATR-based stoploss manages risk. Discrete sizing (0.25) minimizes fee churn.
Target: 12-37 trades/year per symbol (50-150 total over 4 years) for low fee drag and strong generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA20 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous day) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla pivots using previous 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    r4_1d = df_1d_close + 1.5 * range_1d
    s4_1d = df_1d_close - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1w EMA20 for trend filter ===
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter: 20-period average ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        if position == 0:
            # Volume filter: current volume > 2.0x 20-period average (balanced for 6h)
            vol_filter = vol_current > 2.0 * vol_average
            
            # Long conditions: price > R1 (breakout), weekly uptrend, volume filter
            long_breakout = price > r1_1d_aligned[i]
            long_trend = price > ema_20_1w_aligned[i]
            
            # Short conditions: price < S1 (breakdown), weekly downtrend, volume filter
            short_breakout = price < s1_1d_aligned[i]
            short_trend = price < ema_20_1w_aligned[i]
            
            # Entry logic - ONLY enter on volume filter + trend alignment
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below S1 (breakdown)
            elif price < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout)
            elif price > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0