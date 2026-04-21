#!/usr/bin/env python3
"""
1h_Trend_Following_with_Pullback_Entry
Hypothesis: Use 4h EMA50 for trend direction and 1h EMA21 pullback for entry in trending markets.
Add volume confirmation and session filter (08-20 UTC) to reduce noise. Target 15-25 trades/year.
Works in bull/bear by following higher timeframe trend with pullback entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h trend filter: 50-period EMA ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h EMA21 for pullback entry ===
    close = prices['close'].values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_21[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_4h = ema_50_4h_aligned[i]
        ema21_val = ema_21[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Uptrend + pullback to EMA21 + volume confirmation
            if (price_close > trend_4h and      # Uptrend on 4h
                price_close <= ema21_val * 1.01 and  # Near/below EMA21 (pullback)
                price_close >= ema21_val * 0.99 and  # Above/below EMA21 (within 1%)
                vol_spike > 1.5):                 # Volume confirmation
                signals[i] = 0.20
                position = 1
            # Short: Downtrend + pullback to EMA21 + volume confirmation
            elif (price_close < trend_4h and    # Downtrend on 4h
                  price_close >= ema21_val * 0.99 and  # Near/above EMA21 (pullback)
                  price_close <= ema21_val * 1.01 and  # Below/above EMA21 (within 1%)
                  vol_spike > 1.5):                 # Volume confirmation
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit when price crosses EMA21 in opposite direction
            if position == 1 and price_close < ema21_val * 0.99:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema21_val * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Trend_Following_with_Pullback_Entry"
timeframe = "1h"
leverage = 1.0