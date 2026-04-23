#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above R1 AND price > 1w EMA34 AND volume > 1.5x average.
Short when price breaks below S1 AND price < 1w EMA34 AND volume > 1.5x average.
Exit when price reverts to Camarilla Pivot point or volume drops below average.
Camarilla levels provide precise intraday support/resistance. 1w EMA34 ensures trading with higher timeframe trend.
Volume confirmation avoids low-conviction breakouts. Designed for 1d timeframe targeting 30-100 total trades over 4 years.
Works in both bull and bear markets by only taking trades aligned with 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w EMA34 for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start from 2 to have previous day's data
        # Skip if data not ready
        if (i-2 < 0 or np.isnan(ema34_1w_aligned[i]) or np.isnan(high_1d[i-2]) or 
            np.isnan(low_1d[i-2]) or np.isnan(close_1d[i-2])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Previous day's OHLC for Camarilla calculation
        ph = high_1d[i-2]
        pl = low_1d[i-2]
        pc = close_1d[i-2]
        
        # Camarilla levels
        rang = ph - pl
        if rang <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        pivot = (ph + pl + 2 * pc) / 4
        r1 = pivot + rang * 1.1 / 12
        s1 = pivot - rang * 1.1 / 12
        
        # Volume average (20-period)
        if i < 20:
            vol_ma_val = np.mean(volume[max(0, i-19):i+1]) if i > 0 else volume[i]
        else:
            vol_ma_val = np.mean(volume[i-19:i+1])
        
        ema34_val = ema34_1w_aligned[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 1w EMA34 AND volume spike
            if (price > r1 and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND price < 1w EMA34 AND volume spike
            elif (price < s1 and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot OR volume drops below average
                if (price <= pivot or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot OR volume drops below average
                if (price >= pivot or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R1_S1_Breakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0