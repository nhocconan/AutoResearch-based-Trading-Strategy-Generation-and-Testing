#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h/1d trend alignment and volume confirmation
# Long when price breaks above R3 AND 4h close > 4h EMA20 (uptrend) AND 1d close > 1d EMA50 (strong uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below S3 AND 4h close < 4h EMA20 (downtrend) AND 1d close < 1d EMA50 (strong downtrend) AND volume > 1.5x 20 EMA
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-30 trades/year per symbol.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 4h for intermediate trend and 1d for strong trend filter to avoid counter-trend trades.

name = "1h_Camarilla_R3S3_4hEMA20_1dEMA50_VolumeSpike"
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
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h OHLC arrays
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 1d OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 4h bar (using 4h data)
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Align 4h Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Get 4h data for trend filter
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Uptrend when close > EMA20, downtrend when close < EMA20
    uptrend_4h = close_4h > ema_20_4h
    downtrend_4h = close_4h < ema_20_4h
    
    # Align 4h trend to 1h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Get 1d data for strong trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for strong trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Strong uptrend when close > EMA50, strong downtrend when close < EMA50
    strong_uptrend_1d = close_1d > ema_50_1d
    strong_downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d strong trend to 1h timeframe
    strong_uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, strong_uptrend_1d.astype(float))
    strong_downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, strong_downtrend_1d.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or
            np.isnan(strong_uptrend_1d_aligned[i]) or np.isnan(strong_downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 4h uptrend AND 1d strong uptrend AND volume spike
            if (close[i] > r3_aligned[i] and 
                uptrend_4h_aligned[i] > 0.5 and 
                strong_uptrend_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND 4h downtrend AND 1d strong downtrend AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  downtrend_4h_aligned[i] > 0.5 and 
                  strong_downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR 4h trend changes to downtrend OR 1d strong trend changes to downtrend
            if (close[i] < s3_aligned[i] or 
                downtrend_4h_aligned[i] > 0.5 or 
                strong_downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 OR 4h trend changes to uptrend OR 1d strong trend changes to uptrend
            if (close[i] > r3_aligned[i] or 
                uptrend_4h_aligned[i] > 0.5 or 
                strong_uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals