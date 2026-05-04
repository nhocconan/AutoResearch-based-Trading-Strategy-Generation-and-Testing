#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w trend filter and volume confirmation
# Long when price breaks above Camarilla R3 resistance AND weekly bullish trend (close > EMA21) AND volume > 1.5x 20-period volume EMA
# Short when price breaks below Camarilla S3 support AND weekly bearish trend (close < EMA21) AND volume > 1.5x 20-period volume EMA
# Uses weekly EMA21 for trend filter to reduce whipsaw, targeting 10-25 trades/year on 1d.
# Volume confirmation (1.5x) reduces noise trades. Camarilla levels provide precise daily structure from prior day's OHLC.
# Works in bull markets via longs in bullish weekly trend regime and bear markets via shorts in bearish weekly trend regime.

name = "1d_Camarilla_R3S3_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA21 for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_bullish_1w = close_1w > ema_21_1w
    trend_bearish_1w = close_1w < ema_21_1w
    
    # Align weekly trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Get daily data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar using prior day's OHLC
    # We need to align prior day's OHLC to each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 calculation:
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align prior day's Camarilla levels to daily timeframe (wait for day to complete)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND weekly bullish trend AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # Weekly bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND weekly bearish trend AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # Weekly bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR weekly trend turns bearish
            if (close[i] < camarilla_s3_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR weekly trend turns bullish
            if (close[i] > camarilla_r3_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals