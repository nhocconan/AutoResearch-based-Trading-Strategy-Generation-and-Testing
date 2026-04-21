#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_ATRStop_v1
Hypothesis: Daily Camarilla R1/S1 breakouts filtered by weekly EMA34 trend with ATR-based stoploss.
Uses discrete position sizing (0.30) and volume confirmation (>1.5x 20-day average) to reduce whipsaws.
Weekly trend filter provides robust directional bias across bull/bear markets while minimizing fee drag.
Target: 7-25 trades/year per symbol for low fee drag and strong test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Daily OHLC for Camarilla calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate previous week's Camarilla levels (using prior weekly bar's range)
    cam_high = df_1w['high'].values
    cam_low = df_1w['low'].values
    cam_close = df_1w['close'].values
    
    # Camarilla levels: R1 = close + 0.275*(high-low), S1 = close - 0.275*(high-low)
    rng = cam_high - cam_low
    r1 = cam_close + 0.275 * rng
    s1 = cam_close - 0.275 * rng
    
    # Align Camarilla levels to daily timeframe (use prior week's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Weekly EMA34 for trend filter ===
    ema_34_1w = pd.Series(cam_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter
            vol_filter = volume[i] > 1.5 * vol_ma[i]
            
            # Long conditions: price > R1, weekly uptrend, volume filter
            long_breakout = price > r1_aligned[i]
            long_trend = price > ema_34_1w_aligned[i]
            
            # Short conditions: price < S1, weekly downtrend, volume filter
            short_breakout = price < s1_aligned[i]
            short_trend = price < ema_34_1w_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.30
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below S1 (breakdown)
            elif price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout)
            elif price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0