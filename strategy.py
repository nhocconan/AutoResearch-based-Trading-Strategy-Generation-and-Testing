#!/usr/bin/env python3
"""
1d_Pivot_R1S1_Breakout_Volume_Trend
Hypothesis: On daily timeframe, Camarilla pivot levels R1/S1 act as significant support/resistance.
Breakouts above R1 or below S1 with volume confirmation and trend filter (price > EMA50) capture momentum.
Works in bull/bear markets by requiring both breakout and trend alignment, avoiding false breakouts in sideways markets.
Target: 20-50 trades over 4 years (5-12/year) with disciplined entries.
Uses 1d for execution, 1d for pivots and EMA trend filter (self-contained).
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
    
    # === 1d data (primary and HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === Camarilla pivot levels (using previous day's OHLC) ===
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's levels
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        camarilla_r1[i] = c + ((h - l) * 1.1 / 12)
        camarilla_s1[i] = c - ((h - l) * 1.1 / 12)
        camarilla_r3[i] = c + ((h - l) * 1.1 / 4)
        camarilla_s3[i] = c - ((h - l) * 1.1 / 4)
    
    # === EMA50 for trend filter ===
    close_series = pd.Series(close_1d)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume ratio for confirmation ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / vol_ma_20
    
    # Align all data to 1d timeframe (no alignment needed as primary is 1d)
    camarilla_r1_aligned = camarilla_r1
    camarilla_s1_aligned = camarilla_s1
    camarilla_r3_aligned = camarilla_r3
    camarilla_s3_aligned = camarilla_s3
    ema50_aligned = ema50
    vol_ratio_aligned = vol_ratio
    
    signals = np.zeros(n)
    
    # Warmup: enough for EMA50 and volume MA
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema50_val = ema50_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below EMA50 (trend reversal) OR reaches R3 (profit target)
            if price < ema50_val or price > r3:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above EMA50 (trend reversal) OR reaches S3 (profit target)
            if price > ema50_val or price < s3:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above R1 with volume and above EMA50 (uptrend)
            if price > r1 and vol_ratio_val > 1.5 and price > ema50_val:
                signals[i] = 0.30
                position = 1
                continue
            # SHORT: Break below S1 with volume and below EMA50 (downtrend)
            elif price < s1 and vol_ratio_val > 1.5 and price < ema50_val:
                signals[i] = -0.30
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Pivot_R1S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0