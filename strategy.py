#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d Camarilla Pivot Levels (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = pivot + (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    
    s1 = pivot - (range_1d * 1.1 / 12)
    s2 = pivot - (range_1d * 1.1 / 6)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Shift by 1 to use only completed 1d bars (no look-ahead)
    pivot = np.roll(pivot, 1)
    r1 = np.roll(r1, 1)
    r2 = np.roll(r2, 1)
    r3 = np.roll(r3, 1)
    r4 = np.roll(r4, 1)
    s1 = np.roll(s1, 1)
    s2 = np.roll(s2, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    pivot[0] = np.nan
    r1[0] = np.nan
    r2[0] = np.nan
    r3[0] = np.nan
    r4[0] = np.nan
    s1[0] = np.nan
    s2[0] = np.nan
    s3[0] = np.nan
    s4[0] = np.nan
    
    # Align 1d Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 4h EMA(50) > EMA(200) for long, < for short
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up = ema_50 > ema_200
    trend_down = ema_50 < ema_200
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long: price breaks above S3 with volume and trend up
        long_signal = volume_confirmed and trend_up[i] and price_high > s3_aligned[i]
        
        # Short: price breaks below R3 with volume and trend down
        short_signal = volume_confirmed and trend_down[i] and price_low < r3_aligned[i]
        
        # Exit when price returns to pivot level
        exit_long = position == 1 and price_close < pivot_aligned[i]
        exit_short = position == -1 and price_close > pivot_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla pivot breakout strategy on 4h timeframe.
# Uses 1d Camarilla levels (S3/R3) for entry and pivot for exit.
# Enters long when price breaks above S3 with volume confirmation (>1.5x avg volume) and uptrend (EMA50 > EMA200).
# Enters short when price breaks below R3 with volume confirmation and downtrend (EMA50 < EMA200).
# Exits when price returns to the daily pivot level.
# Works in both bull and bear markets by trading breaks of key daily levels with trend and volume filters.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Based on proven Camarilla patterns from top performers in the database.