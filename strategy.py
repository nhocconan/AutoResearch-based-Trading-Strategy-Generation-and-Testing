#!/usr/bin/env python3
"""
6h_1d_Pivot_R3S3_Fade_Reverse_v2
Concept: Fade at R3/S3 with reversal confirmation using RSI divergence and volume exhaustion.
- Short when price touches R3, RSI > 70, and volume < average (exhaustion)
- Long when price touches S3, RSI < 30, and volume < average (exhaustion)
- Exit when price crosses 1-day EMA50 or reverses with opposite RSI extreme
- Designed for mean reversion in ranging markets and exhaustion moves in trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R3S3_Fade_Reverse_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily Camarilla pivots ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # 1-day EMA50 for exit
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h: RSI(14) for divergence/exhaustion ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Get values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema50_val = ema50_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        close_val = prices['close'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(ema50_val) or 
            np.isnan(rsi_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Short: Price touches R3, RSI > 70 (overbought), volume exhaustion
            touch_r3 = abs(close_val - r3_val) / r3_val < 0.002  # Within 0.2%
            rsi_overbought = rsi_val > 70
            vol_exhaustion = vol_ratio_val < 0.8  # Below average volume
            
            if touch_r3 and rsi_overbought and vol_exhaustion:
                signals[i] = -0.25
                position = -1
            # Long: Price touches S3, RSI < 30 (oversold), volume exhaustion
            elif abs(close_val - s3_val) / s3_val < 0.002 and rsi_val < 30 and vol_exhaustion:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Long exit: Price crosses above EMA50 or RSI shows overbought exhaustion
            if close_val > ema50_val or (rsi_val > 70 and abs(close_val - s3_val) / s3_val < 0.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses below EMA50 or RSI shows oversold exhaustion
            if close_val < ema50_val or (rsi_val < 30 and abs(close_val - r3_val) / r3_val < 0.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals