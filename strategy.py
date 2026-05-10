#!/usr/bin/env python3
# 4h_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: 4h breakout above/below daily Camarilla R3/S3 with 12h trend filter and volume confirmation.
# Uses 12h trend (price > 12h EMA50) to filter trades in trending markets, volume surge for confirmation.
# Designed for low trade frequency (<30/year) to avoid fee drag. Works in bull/bear via 12h trend filter.

name = "4h_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily high, low, close for Camarilla levels (R3/S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 = close + 1.1*(high-low)/4
    # Camarilla S3 = close - 1.1*(high-low)/4
    r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align R3 and S3 to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (2.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 12h EMA50
        bullish_trend = close[i] > ema_50_12h_aligned[i]
        bearish_trend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (2.5x average)
        volume_surge = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R3 in bullish trend with volume surge
            if close[i] > r3_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 in bearish trend with volume surge
            elif close[i] < s3_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on opposite signal or trend reversal
            if position == 1:
                # Exit if price breaks below S3 or trend turns bearish
                if close[i] < s3_aligned[i] or not bullish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if price breaks above R3 or trend turns bullish
                if close[i] > r3_aligned[i] or not bearish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals