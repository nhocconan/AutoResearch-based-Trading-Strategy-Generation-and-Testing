#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: On 4h timeframe, use Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
This targets mean-reversion breaks in trending markets, avoiding whipsaw by requiring alignment with daily trend.
Volume spike ensures institutional participation. Designed for 20-50 trades/year to minimize fee drag.
Works in both bull/bear markets via trend filter - only trades in direction of 1d trend.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    # Actually: R4 = C + 1.1*(H-L)/2, R3 = C + 1.1*(H-L)/4
    # S3 = C - 1.1*(H-L)/4, S4 = C - 1.1*(H-L)/2
    # We'll use R3/S3 as entry levels
    
    # Get daily OHLC for previous day
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    H_L = daily_high - daily_low
    R3 = daily_close + 1.1 * H_L / 4
    S3 = daily_close - 1.1 * H_L / 4
    
    # Align daily Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for daily EMA34 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from daily EMA34
        uptrend = close > ema34_1d_aligned[i]
        downtrend = close < ema34_1d_aligned[i]
        
        # Breakout conditions with trend filter
        long_breakout = close[i] > R3_aligned[i]
        short_breakout = close[i] < S3_aligned[i]
        
        # Entry conditions: breakout in direction of trend with volume spike
        long_entry = long_breakout and uptrend and volume_spike[i]
        short_entry = short_breakout and downtrend and volume_spike[i]
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = short_breakout or not uptrend
        short_exit = long_breakout or not downtrend
        
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