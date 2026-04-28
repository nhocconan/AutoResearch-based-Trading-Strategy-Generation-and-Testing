#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: On 12-hour timeframe, enter long when price breaks above Camarilla R3 with volume spike and daily uptrend, short when price breaks below S3 with volume spike and daily downtrend. Exit on opposite breakout (S3 for longs, R3 for shorts) with volume confirmation. Uses Camarilla levels for institutional support/resistance, daily trend filter for trend alignment, and volume spike for institutional confirmation. Designed for low trade frequency (~12-37/year) to minimize fee decay in both bull and bear markets.
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
    
    # Get daily data for Camarilla levels and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    prev_close = df_daily['close'].shift(1).values
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_close + 1.1 * (prev_high - prev_low)
    S3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align daily Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_daily, R3)
    S3_aligned = align_htf_to_ltf(prices, df_daily, S3)
    
    # Daily trend filter: EMA8 > EMA21 for uptrend, EMA8 < EMA21 for downtrend
    close_daily = df_daily['close'].values
    ema8_daily = pd.Series(close_daily).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    ema8_daily_aligned = align_htf_to_ltf(prices, df_daily, ema8_daily)
    ema21_daily_aligned = align_htf_to_ltf(prices, df_daily, ema21_daily)
    
    daily_uptrend = ema8_daily_aligned > ema21_daily_aligned
    daily_downtrend = ema8_daily_aligned < ema21_daily_aligned
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema8_daily_aligned[i]) or np.isnan(ema21_daily_aligned[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume surge
        long_entry = close[i] > R3_aligned[i] and daily_uptrend[i] and volume_surge[i]
        short_entry = close[i] < S3_aligned[i] and daily_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level with volume surge
        long_exit = close[i] < S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > R3_aligned[i] and volume_surge[i]
        
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

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0