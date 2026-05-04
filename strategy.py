#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with volume spike confirmation and 1d trend filter
# Uses 1d Camarilla levels for institutional support/resistance
# Uses 1d EMA50 > EMA200 to ensure bullish trend only (avoid bearish false breakouts)
# Uses 12h volume > 1.5x 20-period EMA for confirmation
# Designed for 12h timeframe targeting 12-37 trades/year with discrete sizing (0.25)
# Volume spike + trend filter reduces false breakouts while maintaining trend alignment
# Works in bull markets (breakouts with volume + trend) and avoids bear markets (no short)

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_200_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 12h volume EMA(20) for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_series = pd.Series(vol_12h)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20)
    
    # Calculate camarilla levels: R3, S3 from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 2
    s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d EMA50 > EMA200 (bullish trend only)
        bullish_trend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        
        # Volume confirmation: current 12h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: close breaks above R3 + volume confirmation + bullish trend
            if (close[i] > r3_aligned[i] and volume_confirmed and 
                bullish_trend):
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit long: price falls below S3 (mean reversion) OR trend turns bearish
            if close[i] < s3_aligned[i] or ema50_1d_aligned[i] <= ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals