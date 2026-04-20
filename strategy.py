#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Breakout with Volume Confirmation and 1d Trend Filter
# - Camarilla levels (R3, S3) from daily OHLC for mean-reversion entries
# - Breakout beyond R4/S4 with volume spike for trend continuation
# - 1d EMA50 filter to align with higher timeframe trend
# - Designed for 6h timeframe to capture both mean reversion and breakout moves
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla levels and EMA filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    rng = high_1d - low_1d
    r4 = close_1d + 1.5 * rng
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike detector (20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 6h price data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if NaN in any indicator
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema50_6h[i]) or np.isnan(vol_ma_20_6h[i]) or
            np.isnan(volume_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ma = vol_ma_20_6h[i]
        ema50 = ema50_6h[i]
        
        # Volume spike condition (at least 1.5x average)
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Mean reversion longs at S3 with volume spike in uptrend
            if price <= s3_6h[i] and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Mean reversion shorts at R3 with volume spike in downtrend
            elif price >= r3_6h[i] and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
            # Breakout longs above R4 with volume spike
            elif price >= r4_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Breakout shorts below S4 with volume spike
            elif price <= s4_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below R3 or volume dries up
            if price < r3_6h[i] or vol < vol_ma * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above S3 or volume dries up
            if price > s3_6h[i] or vol < vol_ma * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_Volume_EMA50Filter"
timeframe = "6h"
leverage = 1.0