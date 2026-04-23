#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and 1d volume confirmation.
Long when price breaks above Camarilla R1 AND 4h EMA50 rising AND 1d volume > 1.2x 20-period MA.
Short when price breaks below Camarilla S1 AND 4h EMA50 falling AND 1d volume > 1.2x 20-period MA.
Exit when price reverts to Camarilla pivot point (PP) or 4h EMA50 reverses.
Uses 4h HTF for trend, 1d for volume filter to avoid low-momentum breakouts.
Camarilla levels provide intraday support/resistance with statistical edge.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivots (based on previous bar's OHLC)
    camarilla_pp = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's OHLC to calculate today's Camarilla levels
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        
        camarilla_pp[i] = (high_prev + low_prev + close_prev) / 3
        camarilla_r1[i] = camarilla_pp[i] + (high_prev - low_prev) * 1.1 / 12
        camarilla_s1[i] = camarilla_pp[i] - (high_prev - low_prev) * 1.1 / 12
    
    # Calculate 1d volume average (20-period) for spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 1)  # EMA50, volume MA, Camarilla (need prev bar)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_pp[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        pp = camarilla_pp[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1h volume > 1.2x 1d volume MA (adaptive to volatility)
        vol_filter = volume[i] > 1.2 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA50 rising AND volume filter
            if price > r1 and ema_rising and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND EMA50 falling AND volume filter
            elif price < s1 and ema_falling and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price returns to pivot point OR EMA50 starts falling
                if price <= pp or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price returns to pivot point OR EMA50 starts rising
                if price >= pp or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_1dVolFilter"
timeframe = "1h"
leverage = 1.0