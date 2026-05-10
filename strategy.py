#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: On 6-hour timeframe, price breaking above Camarilla R3 or below S3 with 1-day trend filter and volume confirmation captures breakout moves in both bull and bear markets. The 1-day trend filter (EMA34) ensures trades align with the dominant daily trend, reducing counter-trend trades. Volume confirmation (volume > 1.5x 20-period average) filters out low-conviction breakouts. Designed for moderate frequency (12-37 trades/year) to balance opportunity and fee drag.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # 1-day data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1-day trend to 6-hour
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Calculate Camarilla levels from previous 1-day bar
    # Camarilla: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1-day bar
    # R4 = close + (high - low) * 1.5
    # R3 = close + (high - low) * 1.25
    # S3 = close - (high - low) * 1.25
    # S4 = close - (high - low) * 1.5
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_prev + rng * 1.25
    camarilla_s3 = close_1d_prev - rng * 1.25
    camarilla_r4 = close_1d_prev + rng * 1.5
    camarilla_s4 = close_1d_prev - rng * 1.5
    
    # Align Camarilla levels to 6-hour
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: break above Camarilla R3 with 1-day uptrend and volume
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_1d_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Camarilla S3 with 1-day downtrend and volume
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_1d_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to Camarilla S3 or trend fails
            if (close[i] < camarilla_s3_aligned[i] or 
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to Camarilla R3 or trend fails
            if (close[i] > camarilla_r3_aligned[i] or 
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals