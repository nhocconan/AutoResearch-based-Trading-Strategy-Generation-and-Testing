#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_ATRStop_v1
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 or below S1 from 12h timeframe indicates institutional breakout, with 12h EMA50 filter for trend alignment and ATR-based stoploss. Uses discrete sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year). Works in both bull and bear markets by capturing breakouts with trend filter and volatility-based risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for Camarilla levels and EMA)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Camarilla Pivot Levels (based on previous bar) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for current bar using previous 12h bar's OHLC
    # Shift by 1 to use previous bar's data (no look-ahead)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # First bar: use same bar's data (will be filtered by min_periods anyway)
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    range_12h = prev_high - prev_low
    camarilla_r1 = prev_close + range_12h * 1.1 / 12
    camarilla_s1 = prev_close - range_12h * 1.1 / 12
    camarilla_r3 = prev_close + range_12h * 1.1 / 4
    camarilla_s3 = prev_close - range_12h * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 12h bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # === 12h EMA50 for trend filter ===
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === ATR for dynamic stoploss and volatility filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema_50 = ema_50_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 + above EMA50
            if price > r1 and price > ema_50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + below EMA50
            elif price < s1 and price < ema_50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # ATR-based stoploss: 2.5 * ATR from entry
            stop_distance = 2.5 * atr_val
            
            if position == 1:
                # Long exit: stoploss hit OR price re-enters R1-S1 range
                if price <= entry_price - stop_distance or price < r1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short exit: stoploss hit OR price re-enters R1-S1 range
                if price >= entry_price + stop_distance or price > s1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0