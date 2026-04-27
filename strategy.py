#!/usr/bin/env python3
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
    
    # Get daily data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivots from previous day's data
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # R2 = Close + 0.5 * (High - Low)
    # R1 = Close + 0.25 * (High - Low)
    # S1 = Close - 0.25 * (High - Low)
    # S2 = Close - 0.5 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    range_1d = high_1d - low_1d
    r4_1d = close_1d + 1.5 * range_1d
    r3_1d = close_1d + 1.0 * range_1d
    s3_1d = close_1d - 1.0 * range_1d
    s4_1d = close_1d - 1.5 * range_1d
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align indicators to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need all indicators
    start_idx = max(14, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        atr_val = atr_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        weekly_trend = ema_21_1w_aligned[i]
        
        # Volatility filter: require minimum volatility
        if i >= 30:
            atr_ma = pd.Series(atr_1d_aligned[:i+1]).rolling(window=30, min_periods=30).mean().iloc[-1]
        else:
            atr_ma = atr_val
        vol_filter = atr_val > (atr_ma * 0.5)
        
        if position == 0:
            # Long conditions: price breaks above R3 with volume, in uptrend
            long_breakout = close[i] > r3
            long_volume = volume[i] > 0  # Volume always positive, placeholder for actual vol filter
            long_trend = close[i] > weekly_trend
            
            if long_breakout and long_volume and long_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short conditions: price breaks below S3 with volume, in downtrend
            elif close[i] < s3 and long_volume and close[i] < weekly_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below S3 (reversal) or hits R4 (take profit)
            if close[i] < s3 or close[i] > r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above R3 (reversal) or hits S4 (take profit)
            if close[i] > r3 or close[i] < s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0