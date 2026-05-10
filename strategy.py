#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout from Camarilla R3/S3 levels with 1d trend filter and volume confirmation.
# Works in bull/bear by following 1d trend direction. Target: 20-40 trades/year per symbol.
# Uses 4h Camarilla levels (R3/S3) from prior day, 1d EMA34 for trend, 4h volume spike.

name = "4H_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for each 4h bar using prior day's OHLC
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We'll use prior day's close, high, low
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data to calculate today's levels
        prev_close = close_1d[i-1]
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        range_hl = prev_high - prev_low
        camarilla_r3[i] = prev_close + range_hl * 1.1 / 4
        camarilla_s3[i] = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Align 1d EMA34 trend to 4h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4h volume spike (volume > 1.5 * 20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema34_1d_aligned[i]
        bearish_trend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long: bullish trend + price breaks above R3 + volume spike
            if bullish_trend and close[i] > r3_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below S3 + volume spike
            elif bearish_trend and close[i] < s3_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price drops below S3
            if bearish_trend or close[i] < s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price rises above R3
            if bullish_trend or close[i] > r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals