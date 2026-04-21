#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_ATRStop_V1
Hypothesis: 12h Camarilla pivot (R1/S1) breakouts filtered by 1w EMA50 trend and ATR-based stoploss.
Enter long when price breaks above 12h R1 with 1w uptrend.
Enter short when price breaks below 12h S1 with 1w downtrend.
Exit on ATR(14) trailing stop (2.5*ATR) or opposite level break.
Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
Works in bull/bear via 1w trend alignment and ATR stoploss.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for pivots, 1w for trend)
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_12h) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 12h Camarilla Pivot Levels (R1, S1) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_12h - low_12h) * 1.1 / 12.0
    r1_12h = close_12h + camarilla_range
    s1_12h = close_12h - camarilla_range
    
    # Align to 12h timeframe (use previous completed 12h bar)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 1w EMA50 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: price > 12h R1, 1w uptrend
            long_breakout = price > r1_12h_aligned[i]
            long_trend = price > ema_50_1w_aligned[i]
            
            # Short conditions: price < 12h S1, 1w downtrend
            short_breakout = price < s1_12h_aligned[i]
            short_trend = price < ema_50_1w_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 12h S1 (support broken)
            elif price < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 12h R1 (resistance broken)
            elif price > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_ATRStop_V1"
timeframe = "12h"
leverage = 1.0