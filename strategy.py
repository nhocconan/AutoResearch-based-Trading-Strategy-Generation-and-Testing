#!/usr/bin/env python3
"""
12h_Weekly_Pullback_Trend
Hypothesis: In strong weekly trends (price above/below weekly EMA50), pullbacks to the 12h EMA21 with volume confirmation offer high-probability entries.
Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
Uses 12h for execution, 1w for trend filter (EMA50), and includes volume confirmation to avoid false signals.
Target: 15-35 trades over 4 years (4-9/year) with disciplined entries to minimize fee drag.
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
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1w data (HTF for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Indicators on 12h ===
    # EMA21 for dynamic support/resistance
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    # Volume 20-period average for confirmation
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    
    # === Weekly EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF data to 12h timeframe
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: enough for EMA50 weekly
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema21_12h_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema21 = ema21_12h_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        weekly_ema50 = ema50_1w_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below EMA21 (trend violation)
            if price < ema21:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above EMA21 (trend violation)
            if price > ema21:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Uptrend: price above weekly EMA50
            if price > weekly_ema50:
                # LONG: Pullback to EMA21 with volume
                if price <= ema21 * 1.005 and vol_ratio > 1.3:  # within 0.5% above EMA21
                    signals[i] = 0.25
                    position = 1
                    continue
            # Downtrend: price below weekly EMA50
            elif price < weekly_ema50:
                # SHORT: Pullback to EMA21 with volume
                if price >= ema21 * 0.995 and vol_ratio > 1.3:  # within 0.5% below EMA21
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Weekly_Pullback_Trend"
timeframe = "12h"
leverage = 1.0