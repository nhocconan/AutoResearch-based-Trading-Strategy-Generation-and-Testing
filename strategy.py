#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Price breaks above Camarilla R3 or below S3 on 12h timeframe with 1d EMA34 trend filter and volume confirmation.
Works in bull markets by capturing upward breakouts above R3 and in bear markets by capturing downward breakouts below S3.
Volume surge confirms institutional participation. Targets 12-37 trades/year to minimize fee drag.
"""

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
    
    # Calculate Camarilla levels on 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align higher timeframe data to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions
        # Long: upward breakout above R3 + uptrend + volume surge
        long_entry = breakout_up and trend_up and volume_surge[i]
        # Short: downward breakout below S3 + downtrend + volume surge
        short_entry = breakout_down and trend_down and volume_surge[i]
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
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

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0