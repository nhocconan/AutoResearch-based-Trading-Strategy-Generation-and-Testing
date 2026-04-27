#!/usr/bin/env python3
"""
#100772 - 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h breakout at Camarilla R3/S3 levels with 1d EMA34 trend filter and volume spike confirmation.
Uses 12h primary timeframe with 1d trend and 1w regime filter to reduce trades and improve quality.
Targets 12-37 trades/year to stay within optimal range for 12h timeframe.
Works in bull (breakouts with trend) and bear (mean reversion from extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for regime filter (choppy vs trending)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly volatility regime using ATR ratio
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[np.inf], tr1])  # First value infinite
    atr10_1w = pd.Series(tr1).rolling(window=10, min_periods=10).mean().values
    atr30_1w = pd.Series(tr1).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr10_1w / atr30_1w
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Calculate Camarilla levels from previous day (to avoid look-ahead)
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    # R3 and S3 levels (wider bands for fewer, higher quality signals)
    daily_r3 = close_1d + daily_range * 1.1 / 4
    daily_s3 = close_1d - daily_range * 1.1 / 4
    daily_pivot_point = (high_1d + low_1d + close_1d) / 3
    
    # Align to 12h timeframe (previous day's levels for current period)
    camarilla_r3 = align_htf_to_ltf(prices, df_1d, daily_r3)
    camarilla_s3 = align_htf_to_ltf(prices, df_1d, daily_s3)
    camarilla_pivot = align_htf_to_ltf(prices, df_1d, daily_pivot_point)
    
    # Volume filter: volume > 2.0x 30-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ATR ratio > 0.8 (avoid extremely choppy markets)
        regime_filter = atr_ratio_aligned[i] > 0.8
        
        # Long condition: price breaks above R3, above 1d EMA34, volume spike, favorable regime
        if (close[i] > camarilla_r3[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i] and 
            regime_filter):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S3, below 1d EMA34, volume spike, favorable regime
        elif (close[i] < camarilla_s3[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i] and 
              regime_filter):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to Camarilla Pivot (mean reversion)
        elif position == 1 and close[i] < camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0