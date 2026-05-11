#!/usr/bin/env python3
name = "4h_TrueRangeBreakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for True Range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily True Range (TR) = max(high-low, high-prev_close, low-prev_close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan  # First day has no previous close
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    
    # True Range is the max of the three components
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR (Average True Range) over 14 days
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Calculate upper and lower bands using ATR multiplier
    # Upper band = previous close + (ATR * multiplier)
    # Lower band = previous close - (ATR * multiplier)
    multiplier = 2.5
    upper_band = prev_close_1d + (atr_14 * multiplier)
    lower_band = prev_close_1d - (atr_14 * multiplier)
    
    # Align bands to 4h timeframe
    upper_band_4h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_4h = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # 4h EMA for trend filter (50-period for smooth trend)
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50).mean().values
    
    # Volume filter: current volume > 2.0x 20-period average (high threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 70
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_band_4h[i]) or np.isnan(lower_band_4h[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper band AND above EMA (uptrend) AND volume spike
            if close[i] > upper_band_4h[i] and close[i] > ema_50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND below EMA (downtrend) AND volume spike
            elif close[i] < lower_band_4h[i] and close[i] < ema_50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below lower band OR below EMA (trend change)
            if close[i] < lower_band_4h[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above upper band OR above EMA (trend change)
            if close[i] > upper_band_4h[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals