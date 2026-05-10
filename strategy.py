#!/usr/bin/env python3
# 4h_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume_Improved
# Hypothesis: 4h breakout of 12h Camarilla R3/S3 levels with 12h trend filter and volume confirmation.
# Uses tighter levels (R3/S3) for stronger signals and 12h trend filter for better trend alignment.
# Designed to work in both bull and bear markets by following the 12h trend direction.
# Expected trade count: ~10-20 per year per symbol to avoid fee drag.

name = "4h_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume_Improved"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for Camarilla levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA for trend filter (34-period)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h ATR for volatility filter (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr_12h = np.maximum(high_12h - low_12h, np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h ATR long-term average for volatility filter (50-period)
    atr_12h_long_avg = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    atr_12h_long_avg_aligned = align_htf_to_ltf(prices, df_12h, atr_12h_long_avg)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_prev = prev_high - prev_low
    s3 = prev_close - 1.1 * range_prev / 4
    r3 = prev_close + 1.1 * range_prev / 4
    
    # Align 12h levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    
    # Volume confirmation (20-period for 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility conditions
    vol_filter = atr_12h_aligned > 0.5 * atr_12h_long_avg_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(s3_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h: close > EMA = uptrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > ema_12h_aligned[i]
        downtrend = close_12h_aligned[i] < ema_12h_aligned[i]
        
        # Volume confirmation (4.0x average for tighter filter)
        volume_surge = volume[i] > 4.0 * vol_ma[i]
        
        # Volatility filter
        if not vol_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R3 in uptrend with volume
            if close[i] > r3_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 in downtrend with volume
            elif close[i] < s3_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close back below R3 or trend fails
                if close[i] < r3_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close back above S3 or trend fails
                if close[i] > s3_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals