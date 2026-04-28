#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeS
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla R3 level and sell when breaks below S3 level, with 12h EMA50 trend filter and volume confirmation. Camarilla levels provide institutional pivot points; breakouts capture momentum in trending markets. Volume surge filters false breakouts. Designed for moderate trade frequency (20-50/year) to balance opportunity and fee decay, working in both bull and bear markets via trend alignment.
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
    
    # Get daily data for Camarilla pivot calculation (using prior day)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior daily bar
    # Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, etc.
    # Using prior day's OHLC to avoid look-ahead
    daily_close = df_daily['close'].values
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Shift to get prior day's values (available at close of current day)
    prior_daily_close = np.roll(daily_close, 1)
    prior_daily_high = np.roll(daily_high, 1)
    prior_daily_low = np.roll(daily_low, 1)
    # First bar has no prior day
    prior_daily_close[0] = daily_close[0]
    prior_daily_high[0] = daily_high[0]
    prior_daily_low[0] = daily_low[0]
    
    # Calculate Camarilla levels for prior day
    daily_range = prior_daily_high - prior_daily_low
    camarilla_r3 = prior_daily_close + daily_range * 1.1 / 4
    camarilla_s3 = prior_daily_close - daily_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_4h = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Trend: bullish when price > EMA50, bearish when price < EMA50
    bullish_trend = close > ema50_12h_4h
    bearish_trend = close < ema50_12h_4h
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or
            np.isnan(ema50_12h_4h[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla breakout with trend and volume
        long_entry = close[i] > camarilla_r3_4h[i] and bullish_trend[i] and volume_surge[i]
        short_entry = close[i] < camarilla_s3_4h[i] and bearish_trend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla breakout with volume (to avoid whipsaw)
        long_exit = close[i] < camarilla_s3_4h[i] and volume_surge[i]
        short_exit = close[i] > camarilla_r3_4h[i] and volume_surge[i]
        
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

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0