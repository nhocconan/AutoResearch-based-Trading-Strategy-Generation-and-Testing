#!/usr/bin/env python3
# 12H_1D_Camarilla_R3_S3_Breakout_Trend_Filter
# Hypothesis: Price breaking Camarilla R3/S3 levels with daily trend alignment and volume confirmation captures strong momentum moves. Works in bull/bear by following daily trend direction. Target: 15-25 trades/year per symbol.

name = "12H_1D_Camarilla_R3_S3_Breakout_Trend_Filter"
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
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    cam_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    cam_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Daily trend filter: EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and trend to 12h
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + daily uptrend + volume spike
            if close[i] > cam_r3_aligned[i] and close[i] > ema34_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + daily downtrend + volume spike
            elif close[i] < cam_s3_aligned[i] and close[i] < ema34_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or trend turns bearish
            if close[i] < cam_s3_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or trend turns bullish
            if close[i] > cam_r3_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals