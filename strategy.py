#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 and close > 1w EMA50 (uptrend) with volume > 2.0x average.
Short when price breaks below S3 and close < 1w EMA50 (downtrend) with volume > 2.0x average.
Uses 4h timeframe to target 75-200 total trades over 4 years. Weekly trend filter ensures alignment
with major market direction, reducing counter-trend trades. Volume spike confirms breakout conviction.
Works in both bull and bear markets by following weekly trend.
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
    
    # Load 1w data for Camarilla pivot calculation and EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels from previous 1w bar
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 4
    # S3 = PP - (H - L) * 1.1 / 4
    pp = (high_1w + low_1w + close_1w) / 3.0
    r3 = pp + (high_1w - low_1w) * 1.1 / 4.0
    s3 = pp - (high_1w - low_1w) * 1.1 / 4.0
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > r3_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < s3_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 OR price breaks below 1w EMA50 (trend reversal)
                if price < s3_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R3 OR price breaks above 1w EMA50 (trend reversal)
                if price > r3_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3_S3_1wEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0