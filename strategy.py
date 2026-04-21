#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeSpike_ATRStop_v1
Hypothesis: 12h Camarilla pivot (R1/S1) breakout filtered by 1w EMA50 trend and volume spike (>2.0x 20-period average).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn.
Weekly trend filter provides robust directional bias across bull/bear markets while reducing whipsaws.
Target: 12-30 trades/year per symbol for low fee drag and strong test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w OHLC for Camarilla pivot calculation (based on previous 1w bar) ===
    df_1w_open = df_1w['open'].values
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1w bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    #            S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low),
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_1w = df_1w_high - df_1w_low
    r1_1w = df_1w_close + 0.275 * range_1w
    s1_1w = df_1w_close - 0.275 * range_1w
    r4_1w = df_1w_close + 1.5 * range_1w
    s4_1w = df_1w_close - 1.5 * range_1w
    
    # Align 1w Camarilla levels to 12h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 1w EMA50 for trend filter ===
    ema_50_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
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
            
            # Long conditions: price > R1 (breakout), 1w uptrend, volume filter
            long_breakout = price > r1_1w_aligned[i]
            long_trend = price > ema_50_1w_aligned[i]
            
            # Short conditions: price < S1 (breakdown), 1w downtrend, volume filter
            short_breakout = price < s1_1w_aligned[i]
            short_trend = price < ema_50_1w_aligned[i]
            
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
            elif price < s1_1w_aligned[i]:
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
            elif price > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0