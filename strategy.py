#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: Camarilla pivot breakouts at R3/S3 levels with 12h trend filter (EMA50) and volume spike confirmation.
# Works in bull/bear by following 12h trend direction. Breakouts are momentum-based and effective in trending regimes.
# Volume spike filters out false breakouts. Target: 20-50 trades/year per symbol.

name = "4H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and volume analysis
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h volume average for spike detection (20-period)
    volume_12h_series = pd.Series(volume_12h)
    vol_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    # R3 = close + 1.1 * (high - low) * 1.1/2
    # S3 = close - 1.1 * (high - low) * 1.1/2
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align 12h trend and volume data to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Volume spike: current volume > 1.5x 12h volume average
        volume_spike = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Enter long: bullish trend + price breaks above Camarilla R3 + volume spike
            if bullish_trend and close[i] > camarilla_r3_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below Camarilla S3 + volume spike
            elif bearish_trend and close[i] < camarilla_s3_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price breaks below Camarilla S3
            if bearish_trend or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price breaks above Camarilla R3
            if bullish_trend or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals