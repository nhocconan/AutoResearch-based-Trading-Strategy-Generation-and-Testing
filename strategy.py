#!/usr/bin/env python3
# 4h_1w_trend_1d_camarilla_v2
# Hypothesis: Trade in direction of weekly trend using daily Camarilla S3/R3 for entry and R4/S4 for exit. Uses volume confirmation to avoid false breaks. Targets 15-35 trades/year (60-140 total) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_trend_1d_camarilla_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    camarilla_S3 = np.zeros(len(high_1d))
    camarilla_R3 = np.zeros(len(high_1d))
    camarilla_S4 = np.zeros(len(high_1d))
    camarilla_R4 = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if range_val > 0:
            camarilla_S3[i] = prev_close - 1.1 * range_val * 1.0/6.0
            camarilla_R3[i] = prev_close + 1.1 * range_val * 1.0/6.0
            camarilla_S4[i] = prev_close - 1.1 * range_val * 1.5/6.0
            camarilla_R4[i] = prev_close + 1.1 * range_val * 1.5/6.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price < S3 (breakdown) or price > R4 (take profit) or weekly trend breaks
            if close[i] < camarilla_S3_aligned[i] or close[i] > camarilla_R4_aligned[i] or close[i] < ema34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R3 (breakout) or price < S4 (take profit) or weekly trend breaks
            if close[i] > camarilla_R3_aligned[i] or close[i] < camarilla_S4_aligned[i] or close[i] > ema34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > S3 with volume surge and weekly uptrend
            if (close[i] > camarilla_S3_aligned[i] and vol_surge and 
                close[i] > ema34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < R3 with volume surge and weekly downtrend
            elif (close[i] < camarilla_R3_aligned[i] and vol_surge and 
                  close[i] < ema34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals