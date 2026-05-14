#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ATRStop_v2
Hypothesis: 4h Camarilla pivot (R1/S1) breakout filtered by 12h EMA50 trend and volume spike (>2.0x 20-period average).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn.
12h trend filter provides robust directional bias across bull/bear markets while reducing whipsaws.
Target: 15-35 trades/year per symbol for low fee drag and strong test generalization.
Enhanced with moderate volume filter and 12h trend confirmation to reduce overtrading vs v1.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA50 trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h OHLC for Camarilla pivot calculation (based on previous 12h bar) ===
    df_12h_open = df_12h['open'].values
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    #            S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low),
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_12h = df_12h_high - df_12h_low
    r1_12h = df_12h_close + 0.275 * range_12h
    s1_12h = df_12h_close - 0.275 * range_12h
    r4_12h = df_12h_close + 1.5 * range_12h
    s4_12h = df_12h_close - 1.5 * range_12h
    
    # Align 12h Camarilla levels to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 12h EMA50 for trend filter ===
    ema_50_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
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
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) 
            or np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        if position == 0:
            # Moderate volume filter: current volume > 2.0x 20-period average
            vol_filter = vol_current > 2.0 * vol_average
            
            # Long conditions: price > R1 (breakout), 12h uptrend, volume filter
            long_breakout = price > r1_12h_aligned[i]
            long_trend = price > ema_50_12h_aligned[i]
            
            # Short conditions: price < S1 (breakdown), 12h downtrend, volume filter
            short_breakout = price < s1_12h_aligned[i]
            short_trend = price < ema_50_12h_aligned[i]
            
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
            elif price < s1_12h_aligned[i]:
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
            elif price > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0