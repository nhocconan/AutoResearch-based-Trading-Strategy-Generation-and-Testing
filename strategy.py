#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: On 12h timeframe, breakout of Camarilla R3/S3 levels with weekly trend filter and volume confirmation captures strong directional moves in both bull and bear markets. Weekly trend avoids counter-trend trades, volume reduces false signals. Designed for low frequency (12-37 trades/year) to minimize fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly trend: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align weekly trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Camarilla levels from previous day (H, L, C)
    # We need previous day's H, L, C to calculate today's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for each day
    # R3 = C + (H-L)*1.1/2
    # S3 = C - (H-L)*1.1/2
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + rng * 1.1 / 2
    camarilla_s3 = close_1d - rng * 1.1 / 2
    
    # Align Camarilla levels to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above Camarilla R3 with weekly uptrend and volume
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_1w_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S3 with weekly downtrend and volume
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_1w_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to Camarilla H3/L3 or trend fails
            # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
            camarilla_h3 = close_1d + rng * 1.1 / 4
            camarilla_l3 = close_1d - rng * 1.1 / 4
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            if (i < len(camarilla_h3_aligned) and i < len(camarilla_l3_aligned) and
                not np.isnan(camarilla_h3_aligned[i]) and not np.isnan(camarilla_l3_aligned[i]) and
                (close[i] < camarilla_h3_aligned[i] or 
                 trend_1w_up_aligned[i] < 0.5)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to Camarilla H3/L3 or trend fails
            camarilla_h3 = close_1d + rng * 1.1 / 4
            camarilla_l3 = close_1d - rng * 1.1 / 4
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            if (i < len(camarilla_h3_aligned) and i < len(camarilla_l3_aligned) and
                not np.isnan(camarilla_h3_aligned[i]) and not np.isnan(camarilla_l3_aligned[i]) and
                (close[i] > camarilla_l3_aligned[i] or 
                 trend_1w_down_aligned[i] < 0.5)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals