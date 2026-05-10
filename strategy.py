#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Price breaks above/below daily Camarilla R3/S3 levels with 1d EMA34 trend filter and volume confirmation.
12h timeframe reduces trade frequency to minimize fee drag. Trend filter ensures alignment with daily momentum,
while volume confirms breakout strength. Works in bull/bear by trading only in direction of 1d trend.
Target: 15-30 trades/year (60-120 total) to stay well within fee limits.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d
    
    # 1d EMA34 for trend filter
    if len(close_1d) >= 34:
        ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema34_1d = np.full(len(close_1d), np.nan)
    
    # Daily volume SMA20 for volume confirmation
    if len(volume_1d) >= 20:
        vol_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    else:
        vol_sma20_1d = np.full(len(volume_1d), np.nan)
    
    # Align all indicators to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average daily volume (scaled)
        # 1d = 2 x 12h bars, so scale daily volume to 12h equivalent
        vol_12h_scaled = vol_sma20_1d_aligned[i] / 2.0  # Average 12h-equivalent volume from 1d data
        volume_confirm = volume[i] > 1.5 * vol_12h_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema34_1d_aligned[i]
        is_downtrend = close[i] < ema34_1d_aligned[i]
        price_above_r3 = close[i] > r3_1d_aligned[i]
        price_below_s3 = close[i] < s3_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, in uptrend, with volume
            if price_above_r3 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, in downtrend, with volume
            elif price_below_s3 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R3 or trend turns down
            if not price_above_r3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S3 or trend turns up
            if not price_below_s3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals