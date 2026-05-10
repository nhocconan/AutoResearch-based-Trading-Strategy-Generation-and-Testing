#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot breakout with volume confirmation and 1-day EMA trend filter.
# Works in bull/bear markets: Camarilla levels provide institutional support/resistance,
# volume confirms breakout strength, and 1-day trend ensures alignment with higher timeframe momentum.
# Target: 25-40 trades/year to avoid fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Camarilla levels from previous day
    # We'll use daily high/low/close from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, R3, S3, S3 levels
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    cam_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    cam_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # 1-day trend filter (EMA34)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1-day trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume and 1-day uptrend
            if (close[i] > cam_r3_aligned[i] and
                volume_confirmed[i] and
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with volume and 1-day downtrend
            elif (close[i] < cam_s3_aligned[i] and
                  volume_confirmed[i] and
                  trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below Camarilla S3 or 1-day trend turns down
            if (close[i] < cam_s3_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above Camarilla R3 or 1-day trend turns up
            if (close[i] > cam_r3_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals