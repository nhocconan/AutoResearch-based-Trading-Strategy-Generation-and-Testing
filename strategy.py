#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 breakouts on 12h with 1d trend filter (EMA34) and volume confirmation.
# Works in bull: breakouts capture momentum. Works in bear: 1d EMA filter avoids counter-trend trades.
# Target: 20-50 trades/year, low frequency to minimize fee drag.
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need previous day's data, so shift by 1
    if len(close_1d) < 2:
        return np.zeros(n)
    prev_close = close_1d[:-1]
    prev_high = high_1d[:-1]
    prev_low = low_1d[:-1]
    # Calculate for each previous day
    camarilla_R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    # Now shift forward to align with current day (today's levels based on yesterday)
    camarilla_R3 = np.concatenate([ [np.nan], camarilla_R3 ])
    camarilla_S3 = np.concatenate([ [np.nan], camarilla_S3 ])
    
    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d data to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average volume
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > avg_volume * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 40  # need EMA34 and Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + above 1d EMA34 + volume confirmation
            if close[i] > camarilla_R3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + below 1d EMA34 + volume confirmation
            elif close[i] < camarilla_S3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (reversal) or below 1d EMA34 (trend change)
            if close[i] < camarilla_S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 (reversal) or above 1d EMA34 (trend change)
            if close[i] > camarilla_R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals