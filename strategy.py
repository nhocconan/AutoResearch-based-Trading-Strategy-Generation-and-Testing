#!/usr/bin/env python3
# 12H_1D_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Price breaking Camarilla R3/S3 levels on 12h with volume confirmation and daily trend filter captures strong moves in both bull and bear markets.
# Uses daily EMA50 for trend direction and Camarilla levels from prior daily close for breakout signals.
# Volume filter ensures breakouts have conviction. Designed for low trade frequency (<30/year) to minimize fee drag.

name = "12H_1D_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'.encode() if isinstance('high', str) else 'high'] if isinstance(prices, dict) else prices['high'].values
    low = prices['low'.encode() if isinstance('low', str) else 'low'] if isinstance(prices, dict) else prices['low'].values
    volume = prices['volume'.encode() if isinstance('volume', str) else 'volume'] if isinstance(prices, dict) else prices['volume'].values
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous daily bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Align trend filter
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Volume filter: 12h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + price breaks above R3 + volume confirmation
            if bullish and high[i] > camarilla_r3_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below S3 + volume confirmation
            elif bearish and low[i] < camarilla_s3_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price breaks below S3 (reversal)
            if bearish or low[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price breaks above R3 (reversal)
            if bullish or high[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals