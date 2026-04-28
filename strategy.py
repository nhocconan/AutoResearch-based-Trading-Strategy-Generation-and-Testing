#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeS
Hypothesis: On 4h timeframe, use Camarilla R3/S3 levels from 1d timeframe for breakout entries, filtered by 12h EMA trend and volume confirmation. Camarilla levels provide key support/resistance, trend filter avoids counter-trend trades, and volume confirms institutional participation. Designed for 20-40 trades/year to minimize fee drag and work in both bull/bear markets via trend alignment.
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
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_close = np.roll(df_daily['close'].values, 1)
    prev_high = np.roll(df_daily['high'].values, 1)
    prev_low = np.roll(df_daily['low'].values, 1)
    # First day has no previous, use current values
    prev_close[0] = df_daily['close'].values[0]
    prev_high[0] = df_daily['high'].values[0]
    prev_low[0] = df_daily['low'].values[0]
    
    camarilla_R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_S3)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h EMA21 for trend filter
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Trend: bullish when price > EMA21, bearish when price < EMA21
    bullish_trend = close > ema21_12h_aligned
    bearish_trend = close < ema21_12h_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Wait for EMA21 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema21_12h_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        long_entry = close[i] > camarilla_R3_aligned[i] and bullish_trend[i] and volume_surge[i]
        short_entry = close[i] < camarilla_S3_aligned[i] and bearish_trend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level with volume surge
        long_exit = close[i] < camarilla_S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > camarilla_R3_aligned[i] and volume_surge[i]
        
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