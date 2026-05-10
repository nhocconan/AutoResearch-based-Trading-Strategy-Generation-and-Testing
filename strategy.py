#!/usr/bin/env python3
# 12h_1d_Camarilla_R3_S3_Breakout_1wTrend_Volume_v2
# Hypothesis: 12h breakout of daily Camarilla R3/S3 levels with weekly trend filter and volume confirmation.
# Uses weekly EMA trend filter to avoid counter-trend trades in sideways markets.
# Volume confirmation (2.0x 20-period average) ensures breakouts have participation.
# Designed for fewer, higher-quality trades (target: 50-150 total over 4 years).
# Discrete position sizing (0.25) minimizes churn and manages drawdown.
# Works in bull markets via trend-following breakouts and in bear via counter-trend reversals at S3/R3.

name = "12h_1d_Camarilla_R3_S3_Breakout_1wTrend_Volume_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily and weekly data (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA for trend filter (21-period)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_prev = prev_high - prev_low
    s3 = prev_close - 1.1 * range_prev / 4
    r3 = prev_close + 1.1 * range_prev / 4
    
    # Align daily levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume confirmation (20-period for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility conditions
    vol_filter = atr_1d_aligned > 0.5 * pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(s3_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from weekly: close > EMA = uptrend
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        uptrend = close_1w_aligned[i] > ema_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_1w_aligned[i]
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
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