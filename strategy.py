#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R1 AND 4h close > 4h EMA50 (uptrend) AND volume > 1.8x 20 EMA
# Short when price breaks below S1 AND 4h close < 4h EMA50 (downtrend) AND volume > 1.8x 20 EMA
# Uses 1h for primary signals, 4h for trend to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise trades. Discrete sizing (0.20) minimizes fee churn.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h. Works in both bull and bear via directional alignment.

name = "1h_Camarilla_R1S1_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Camarilla levels (based on previous hour's OHLC)
    # We need hourly OHLC for Camarilla calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Get hourly OHLC arrays
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Camarilla levels for each hour
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_r1 = close_1h + (high_1h - low_1h) * 1.1 / 12
    camarilla_s1 = close_1h - (high_1h - low_1h) * 1.1 / 12
    
    # Align hourly Camarilla levels to 1h timeframe (no shift needed as already aligned)
    r1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)
    
    # Get 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_4h = close_4h > ema_50_4h
    downtrend_4h = close_4h < ema_50_4h
    
    # Align 4h trend to 1h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND 4h uptrend AND volume spike
            if (close[i] > r1_aligned[i] and 
                uptrend_4h_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S1 AND 4h downtrend AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  downtrend_4h_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR 4h trend changes to downtrend
            if (close[i] < s1_aligned[i] or 
                downtrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R1 OR 4h trend changes to uptrend
            if (close[i] > r1_aligned[i] or 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals