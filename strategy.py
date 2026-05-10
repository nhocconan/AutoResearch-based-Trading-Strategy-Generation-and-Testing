#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: 4H Camarilla R3/S3 breakout with 1-day EMA34 trend filter and volume spike captures institutional breakouts in both bull and bear markets. Camarilla levels act as intraday support/resistance, EMA34 filters trend direction, volume spike confirms institutional participation. Designed for 20-40 trades/year to minimize fee drag while capturing sustained moves.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # 1-day data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1D bar: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually standard Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # We'll use R3 and S3 as breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1D indicators to 4H
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 20-period average on 4H
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0  # Require strong volume confirmation
        
        if position == 0:
            # Enter long: break above Camarilla R3 with 1D uptrend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_1d_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Camarilla S3 with 1D downtrend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_1d_down_aligned[i] > 0.5 and volume_spike):
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