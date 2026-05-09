#!/usr/bin/env python3
# 6H_1D_1W_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: On 6h timeframe, enter long when price breaks above weekly Camarilla R3 level with 1d uptrend and volume confirmation.
# Short when price breaks below weekly Camarilla S3 level with 1d downtrend and volume confirmation.
# Uses weekly Camarilla levels for structure, 1d trend filter to avoid counter-trend trades, and volume for confirmation.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).

name = "6H_1D_1W_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels: R3, S3 based on previous week
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    # Camarilla R3 = close + (range * 1.1/4)
    # Camarilla S3 = close - (range * 1.1/4)
    camarilla_r3 = close_1w + (range_1w * 1.1 / 4)
    camarilla_s3 = close_1w - (range_1w * 1.1 / 4)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly Camarilla levels to 6h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Align daily trend to 6h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Camarilla R3 + 1d uptrend + volume confirmation
            if close[i] > camarilla_r3_aligned[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Camarilla S3 + 1d downtrend + volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Camarilla S3 (reversal) or trend changes
            if close[i] < camarilla_s3_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Camarilla R3 (reversal) or trend changes
            if close[i] > camarilla_r3_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals