#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Use daily Camarilla pivot levels R3/S3 for 6h breakout entries, filtered by weekly trend and volume spike.
Camarilla levels identify key intraday support/resistance where breakouts often continue in trend direction.
Weekly EMA filter avoids counter-trend trades; volume spike confirms breakout strength.
Designed for 15-25 trades/year per symbol, works in bull/bear via trend filter.
"""

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day)
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Typical price for Camarilla
    typical_price = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels: R3/S3 are most significant for breakouts
    # R3 = close + 1.1 * (high - low) * 1.1/4
    # S3 = close - 1.1 * (high - low) * 1.1/4
    camarilla_r3 = close_prev + 1.1 * range_prev * 1.1 / 4
    camarilla_s3 = close_prev - 1.1 * range_prev * 1.1 / 4
    
    # Align daily Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    
    # Weekly EMA20 for trend filter
    ema_20 = pd.Series(df_weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_weekly, ema_20)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 24-period EMA (48h average)
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_filter = volume > vol_ema24 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla (1 day), weekly EMA20 (20 weeks), volume EMA (24)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA20 (uptrend) AND price breaks above Camarilla R3 with volume spike
            if close[i] > ema_20_aligned[i] and high[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA20 (downtrend) AND price breaks below Camarilla S3 with volume spike
            elif close[i] < ema_20_aligned[i] and low[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla pivot point (PP) OR trend turns bearish
            # Calculate Camarilla PP for exit: (high+low+close)/3
            typical_prev = (high_prev[i] + low_prev[i] + close_prev[i]) / 3 if i >= 1 else camarilla_r3_aligned[i] - (camarilla_r3_aligned[i] - camarilla_s3_aligned[i]) / 2
            if low[i] < typical_prev or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla pivot point (PP) OR trend turns bullish
            typical_prev = (high_prev[i] + low_prev[i] + close_prev[i]) / 3 if i >= 1 else camarilla_r3_aligned[i] - (camarilla_r3_aligned[i] - camarilla_s3_aligned[i]) / 2
            if high[i] > typical_prev or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals