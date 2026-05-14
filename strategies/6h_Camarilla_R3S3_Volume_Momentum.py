#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot + 1d Volume Spike + Momentum Confirmation
# - Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period average + RSI(14) > 50
# - Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period average + RSI(14) < 50
# - Exit when price crosses back through Camarilla R2/S2 levels or momentum reverses
# - Uses daily pivots for structure and volume spike for confirmation
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    r3_1d = pivot_1d + range_1d * 1.1 / 2
    r2_1d = pivot_1d + range_1d * 1.1 / 4
    r1_1d = pivot_1d + range_1d * 1.1 / 6
    s1_1d = pivot_1d - range_1d * 1.1 / 6
    s2_1d = pivot_1d - range_1d * 1.1 / 4
    s3_1d = pivot_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate RSI(14) on 6h timeframe
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after RSI/volume warmup
        # Skip if NaN in indicators
        if np.isnan(r3_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or \
           np.isnan(s2_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + volume spike + bullish momentum
            if price > r3_1d_aligned[i] and vol > 2.0 * vol_ma[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + volume spike + bearish momentum
            elif price < s3_1d_aligned[i] and vol > 2.0 * vol_ma[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below R2 or momentum turns bearish
            if price < r2_1d_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above S2 or momentum turns bullish
            if price > s2_1d_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Volume_Momentum"
timeframe = "6h"
leverage = 1.0