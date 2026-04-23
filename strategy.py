#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter, volume spike, and session filter (08-20 UTC).
Long when price breaks above Camarilla R1 AND price > 4h EMA50 AND volume > 1.8x 20-period average AND within session.
Short when price breaks below Camarilla S1 AND price < 4h EMA50 AND volume > 1.8x 20-period average AND within session.
Exit when price reverts to Camarilla pivot (PP) OR ATR trailing stop (1.5*ATR from extreme).
Uses 4h HTF for trend alignment and daily for Camarilla levels.
Target: 15-35 trades/year on 1h timeframe with discrete sizing 0.20.
Works in bull via breakouts, in bear via mean reversion to PP and volatility filters.
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
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous day (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: PP = (H+L+C)/3, Range = H-L
    pp = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    # R1 = PP + 1.1 * Range / 4, S1 = PP - 1.1 * Range / 4
    r1 = pp + 1.1 * rng / 4.0
    s1 = pp - 1.1 * rng / 4.0
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # 1h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 1h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # vol_ma20, ema_50_4h
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_val = ema_50_4h_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0 and in_session:
            # Long: price breaks above R1 AND price > 4h EMA50 AND volume spike
            if price > r1_val and price > ema_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: price breaks below S1 AND price < 4h EMA50 AND volume spike
            elif price < s1_val and price < ema_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to pivot point (PP)
            if position == 1 and price < pp_val:
                exit_signal = True
            elif position == -1 and price > pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 1.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 1.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 1.5 * atr_val:
                exit_signal = True
            
            # Time-based exit: close position if outside session
            if not in_session:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter_PPExit_ATRTrailingStop"
timeframe = "1h"
leverage = 1.0