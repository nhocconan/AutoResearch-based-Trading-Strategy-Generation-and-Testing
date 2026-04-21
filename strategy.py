#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ATR_Stop_v2
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 or below S1 from 1d timeframe indicates institutional breakout, with 1d EMA34 filter for trend alignment and ATR-based stoploss for risk control. Uses discrete sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels and EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar: use same day's data (will be filtered by min_periods anyway)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + range_1d * 1.1 / 12
    camarilla_s1 = prev_close - range_1d * 1.1 / 12
    camarilla_r3 = prev_close + range_1d * 1.1 / 4
    camarilla_s3 = prev_close - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h ATR(14) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema_34 = ema_34_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 + above EMA34
            if price > r1 and price > ema_34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + below EMA34
            elif price < s1 and price < ema_34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Check stoploss: 2 * ATR from entry
            if position == 1:
                stop_price = entry_price - 2.0 * atr_val
                # Exit conditions: stoploss hit OR price re-enters R1-S1 range
                if price <= stop_price or price < r1:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                stop_price = entry_price + 2.0 * atr_val
                # Exit conditions: stoploss hit OR price re-enters R1-S1 range
                if price >= stop_price or price > s1:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ATR_Stop_v2"
timeframe = "4h"
leverage = 1.0