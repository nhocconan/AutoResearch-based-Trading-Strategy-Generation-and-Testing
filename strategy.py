#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation (>1.8x 20 EMA volume)
# Uses tighter 12h Camarilla pivot levels (R3/S3) for higher-probability breakouts with less noise
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws in bear markets
# Volume confirmation filters false breakouts (>1.8x average volume) to reduce trade frequency
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 80-180 total trades over 4 years = 20-45/year for 12h timeframe (avoiding overtrading)
# Works in bull markets (continuation at R4/S4) and bear markets (continuation at R3/S3)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_Camarilla_R3S3_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot points (based on prior completed 1d bar)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1/4
    # S3 = Pivot - Range * 1.1/4
    # R4 = Pivot + Range * 1.1/2
    # S4 = Pivot - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r3_1d = pivot_1d + (range_1d * 1.1 / 4.0)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4.0)
    r4_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    r3_1d_shifted = np.roll(r3_1d, 1)
    s3_1d_shifted = np.roll(s3_1d, 1)
    r4_1d_shifted = np.roll(r4_1d, 1)
    s4_1d_shifted = np.roll(s4_1d, 1)
    r3_1d_shifted[0] = np.nan
    s3_1d_shifted[0] = np.nan
    r4_1d_shifted[0] = np.nan
    s4_1d_shifted[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_shifted)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d_shifted)
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 1d EMA50 AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 1d EMA50 AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1d EMA50
            if close[i] < s3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1d EMA50
            if close[i] > r3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals