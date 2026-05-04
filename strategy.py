#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 Breakout with 12h EMA50 Trend Filter and Volume Spike Confirmation
# Camarilla levels provide institutional support/resistance. Breakout above R3 or below S3 with volume confirms institutional participation.
# 12h EMA50 ensures alignment with medium-term trend to avoid counter-trend trades.
# Designed for 20-50 trades/year on 4h to minimize fee drag while capturing strong trending moves.
# Works in bull markets via long R3 breakouts in uptrend and in bear markets via short S3 breakdowns in downtrend.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    # Range = High - Low
    rng = high_1d - low_1d
    # Camarilla levels
    # R4 = Close + Range * 1.5000
    # R3 = Close + Range * 1.2500
    # R2 = Close + Range * 1.1666
    # R1 = Close + Range * 1.0833
    # S1 = Close - Range * 1.0833
    # S2 = Close - Range * 1.1666
    # S3 = Close - Range * 1.2500
    # S4 = Close - Range * 1.5000
    r3 = close_1d + rng * 1.2500
    s3 = close_1d - rng * 1.2500
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 12h uptrend AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and  # 12h uptrend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below S3 AND 12h downtrend AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and  # 12h downtrend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below pivot OR 12h trend turns down
            # Pivot for exit: use same day's pivot as reference
            pivot_1d = (high_1d + low_1d + close_1d) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if close[i] < pivot_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above pivot OR 12h trend turns up
            pivot_1d = (high_1d + low_1d + close_1d) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if close[i] > pivot_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals