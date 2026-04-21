#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from 12h timeframe provide institutional support/resistance.
Breakout above R1 or below S1 with volume confirmation and 12h EMA50 trend filter captures
institutional moves. Designed for moderate trade frequency (target 25-50 trades/year) to
balance signal quality with fee minimization. Works in both bull and bear markets by
following the 12h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === Calculate Camarilla pivot levels (R1, S1) on 12h ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    r1_12h = close_12h + range_12h * 1.1 / 12
    s1_12h = close_12h - range_12h * 1.1 / 12
    
    # === 12h EMA50 trend filter ===
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 12h Volume average (20-period) for spike detection ===
    volume_12h = df_12h['volume'].values
    vol_avg_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 12h indicators to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(r1_12h_aligned[i]) or
            np.isnan(s1_12h_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        volume_current = prices['volume'].iloc[i]
        r1 = r1_12h_aligned[i]
        s1 = s1_12h_aligned[i]
        ema_50 = ema_50_12h_aligned[i]
        vol_avg = vol_avg_20_aligned[i]
        
        # Volume spike condition (2x average volume)
        volume_spike = volume_current > 2.0 * vol_avg
        
        if position == 0:
            # Long: Price breaks above R1, above 12h EMA50 (uptrend), with volume spike
            if (price_close > r1 and 
                price_close > ema_50 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, below 12h EMA50 (downtrend), with volume spike
            elif (price_close < s1 and 
                  price_close < ema_50 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses back through the pivot level in opposite direction
            if position == 1 and price_close < r1:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > s1:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0