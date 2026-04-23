#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 (1h) AND price > 4h EMA50 (uptrend) AND volume > 1.5x average.
Short when price breaks below Camarilla S3 (1h) AND price < 4h EMA50 (downtrend) AND volume > 1.5x average.
Exit when price reverts to Camarilla pivot point (PP) or trend reverses (price crosses 4h EMA50).
Uses 1h primary timeframe with 4h HTF for trend direction to balance trade frequency (~20-40/year) and capture institutional breakout sessions.
Works in both bull and bear markets by requiring 4h EMA50 trend confirmation for breakout entries.
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
    
    # Load 1h data for Camarilla pivot - ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Camarilla pivot levels for 1h timeframe
    typical_price = (high_1h + low_1h + close_1h) / 3
    price_range = high_1h - low_1h
    camarilla_pp = typical_price
    camarilla_r3 = camarilla_pp + price_range * 1.1 / 2
    camarilla_s3 = camarilla_pp - price_range * 1.1 / 2
    
    # Align 1h Camarilla levels to 1h timeframe (no shift needed as same TF)
    camarilla_pp_aligned = camarilla_pp
    camarilla_r3_aligned = camarilla_r3
    camarilla_s3_aligned = camarilla_s3
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = camarilla_pp_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        ema50_val = ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 AND price > 4h EMA50 (uptrend) AND volume spike
            if (price > r3_val and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND price < 4h EMA50 (downtrend) AND volume spike
            elif (price < s3_val and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot point OR price breaks below 4h EMA50 (trend reversal)
                if price <= pp_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot point OR price breaks above 4h EMA50 (trend reversal)
                if price >= pp_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3_S3_4hEMA50_Volume_Breakout"
timeframe = "1h"
leverage = 1.0