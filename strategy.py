#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation.
Long when price breaks above R1 (4h Camarilla) AND price > 4h EMA20 (uptrend) AND volume > 2x average.
Short when price breaks below S1 (4h Camarilla) AND price < 4h EMA20 (downtrend) AND volume > 2x average.
Exit when price reverts to Camarilla pivot point (PP) or trend reverses (price crosses 4h EMA20).
Uses 1h timeframe for entry timing, with 4h for signal direction to target 15-30 trades/year.
Session filter (08-20 UTC) reduces noise. Position size 0.20 to manage drawdown.
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
    
    # Load 4h data for Camarilla pivot levels and EMA20 trend - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Camarilla pivot levels for 4h
    camarilla_pp = (high_4h + low_4h + close_4h) / 3.0
    camarilla_r1 = close_4h + (high_4h - low_4h) * 1.1 / 12.0
    camarilla_s1 = close_4h - (high_4h - low_4h) * 1.1 / 12.0
    
    # Calculate EMA20 for 4h trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_val = ema20_4h_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND price > 4h EMA20 (uptrend) AND volume > 2x average
            if (price > r1_val and price > ema20_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND price < 4h EMA20 (downtrend) AND volume > 2x average
            elif (price < s1_val and price < ema20_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot point OR price breaks below 4h EMA20 (trend reversal)
                if price <= pp_val or price < ema20_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot point OR price breaks above 4h EMA20 (trend reversal)
                if price >= pp_val or price > ema20_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1_S1_Breakout_4hEMA20_Volume_Session"
timeframe = "1h"
leverage = 1.0