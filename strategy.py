#!/usr/bin/env python3
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
    
    # === Daily Camarilla Pivot Levels (from previous day) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels from previous day (shifted by 1)
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.083
    s1 = pp - (high_1d - low_1d) * 1.083
    r2 = pp + (high_1d - low_1d) * 1.166
    s2 = pp - (high_1d - low_1d) * 1.166
    r3 = pp + (high_1d - low_1d) * 1.250
    s3 = pp - (high_1d - low_1d) * 1.250
    r4 = pp + (high_1d - low_1d) * 1.500
    s4 = pp - (high_1d - low_1d) * 1.500
    
    # Align to 6h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h ATR for volatility filter ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume Spike Detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)  # Strong volume spike
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # === EXIT LOGIC: Exit when price touches opposite S1/R1 or volatility drops ===
        if position == 1:  # Long position
            # Exit when price touches S1 or volatility drops significantly
            if price <= s1_aligned[i] or atr[i] < (atr[i-1] * 0.6 if i > warmup else atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches R1 or volatility drops significantly
            if price >= r1_aligned[i] or atr[i] < (atr[i-1] * 0.6 if i > warmup else atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above R1 with volume spike (continuation)
            if price > r1_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Break below S1 with volume spike (continuation)
            elif price < s1_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                continue
            
            # LONG: Mean reversion from S3 (strong oversold)
            elif price <= s3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Mean reversion from R3 (strong overbought)
            elif price >= r3_aligned[i] and volume_spike[i]:
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

name = "6h_Camarilla_S1S3_R1R3_Breakout_MeanReversion"
timeframe = "6h"
leverage = 1.0