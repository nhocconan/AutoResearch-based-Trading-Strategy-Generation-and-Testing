#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: On 12-hour timeframe, use weekly trend filter (price vs 200 EMA) and Camarilla pivot levels (R3/S3) from daily timeframe for breakout entries. Enter long when price breaks above R3 with volume confirmation in weekly uptrend, short when price breaks below S3 with volume confirmation in weekly downtrend. Exit when price returns to the daily Pivot point. Designed for low-frequency, high-conviction trades to avoid fee drag in choppy markets like 2025.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Load weekly data for trend filter (200 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load daily data for Camarilla pivot levels (R3, S3, Pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 12-hour timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume ratio: current volume / 50-period average volume (on 12h)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for 200 EMA and other indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        weekly_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation: volume > 2.0x average to filter weak breakouts
        volume_confirm = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above R3 in weekly uptrend with volume
            long_entry = (close[i] > r3_aligned[i]) and weekly_uptrend and volume_confirm
            # Short: price breaks below S3 in weekly downtrend with volume
            short_entry = (close[i] < s3_aligned[i]) and weekly_downtrend and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below daily Pivot point
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above daily Pivot point
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals