#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and session (08-20 UTC)
# Uses 4h EMA200 for trend direction, 1h for precise entry timing on Camarilla levels
# Volume confirmation reduces false breakouts. Discrete sizing (0.20) controls drawdown.
# Designed to work in both bull (trend follow) and bear (mean revert at extremes) via regime.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA200 for trend filter
    ema_200_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Get 1d HTF data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1, R2, S2, R3, S3)
    # Using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_1d * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12)
    r2 = pp + (range_1d * 1.1 / 6)
    s2 = pp - (range_1d * 1.1 / 6)
    r3 = pp + (range_1d * 1.1 / 4)
    s3 = pp - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1h volume average (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for EMA200
        # Skip if any required data is NaN
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine trend bias from 4h EMA200
        bullish_bias = close[i] > ema_200_4h_aligned[i]
        bearish_bias = close[i] < ema_200_4h_aligned[i]
        
        # Initialize signal
        signal_val = 0.0
        
        # Long conditions: price breaks above Camarilla resistance with volume
        if bullish_bias and vol_confirm:
            if close[i] > r1_aligned[i]:
                signal_val = 0.20  # Long 20%
            elif close[i] > r2_aligned[i]:
                signal_val = 0.20  # Long 20% (same size for simplicity)
            elif close[i] > r3_aligned[i]:
                signal_val = 0.20  # Long 20%
        
        # Short conditions: price breaks below Camarilla support with volume
        elif bearish_bias and vol_confirm:
            if close[i] < s1_aligned[i]:
                signal_val = -0.20  # Short 20%
            elif close[i] < s2_aligned[i]:
                signal_val = -0.20  # Short 20%
            elif close[i] < s3_aligned[i]:
                signal_val = -0.20  # Short 20%
        
        signals[i] = signal_val
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA200_Vol_Session_v1"
timeframe = "1h"
leverage = 1.0