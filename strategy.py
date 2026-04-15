#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band Squeeze with 1-week Trend Filter and Volume Confirmation
# Uses Bollinger Band width to identify low volatility periods (squeeze) and breaks
# in the direction of the weekly trend (EMA50). Volume confirmation ensures breakout
# validity. Works in both bull and bear markets by following the higher timeframe trend.
# Target: 20-50 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Bollinger Bands (20, 2) on daily
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band Squeeze: width below 20-period mean
    bb_width_mean = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_mean
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema_50_1w
    weekly_downtrend = close_1w < ema_50_1w
    
    # Align indicators to 1d timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            continue
        
        # Long entry: Bollinger breakout up + weekly uptrend + volume
        if (squeeze_aligned[i] and
            close[i] > upper_bb[i] and
            weekly_uptrend_aligned[i] > 0.5 and
            volume[i] > 1.5 * vol_ma_20[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bollinger breakout down + weekly downtrend + volume
        elif (squeeze_aligned[i] and
              close[i] < lower_bb[i] and
              weekly_downtrend_aligned[i] > 0.5 and
              volume[i] > 1.5 * vol_ma_20[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Bollinger band touch or loss of squeeze
        elif position == 1 and (close[i] < lower_bb[i] or not squeeze_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > upper_bb[i] or not squeeze_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Squeeze_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0