#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Price breaking through Camarilla R3/S3 levels on 4h timeframe with 1-day EMA34 trend filter and volume spike confirmation. Camarilla levels derived from prior day's range provide institutional support/resistance. Works in bull markets by capturing R3 breakouts (bullish continuation) and in bear markets by capturing S3 breakdowns (bearish continuation). Volume surge confirms institutional participation. Targets 20-40 trades/year to minimize fee drag.
"""

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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    high_prev = df_1d['high'].shift(1).values  # Previous day high
    low_prev = df_1d['low'].shift(1).values    # Previous day low
    close_prev = df_1d['close'].shift(1).values # Previous day close
    
    range_prev = high_prev - low_prev
    camarilla_r3 = close_prev + (range_prev * 1.1 / 6)  # R3 level
    camarilla_s3 = close_prev - (range_prev * 1.1 / 6)  # S3 level
    
    # Get 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align higher timeframe data to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_r3 = close[i] > camarilla_r3_aligned[i]
        breakdown_s3 = close[i] < camarilla_s3_aligned[i]
        
        # Trend filter: price above/below 1-day EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions
        # Long: R3 breakout + uptrend + volume spike
        long_entry = breakout_r3 and trend_up and volume_spike[i]
        # Short: S3 breakdown + downtrend + volume spike
        short_entry = breakdown_s3 and trend_down and volume_spike[i]
        
        # Exit conditions: opposite level break or trend reversal
        long_exit = breakdown_s3 or not trend_up
        short_exit = breakout_r3 or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0