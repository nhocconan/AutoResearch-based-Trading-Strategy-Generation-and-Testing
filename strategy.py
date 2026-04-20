#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis (price action, volume, volatility)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Price Range (high - low) for volatility assessment
    daily_range = high_1d - low_1d
    range_ma_20 = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    range_ratio = daily_range / np.where(range_ma_20 == 0, 1, range_ma_20)
    range_ratio_aligned = align_htf_to_ltf(prices, df_1d, range_ratio)
    
    # 1d Volume ratio (current volume / 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # 1d Close price for trend context
    close_1d_ma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    close_1d_ma_50_aligned = align_htf_to_ltf(prices, df_1d, close_1d_ma_50)
    
    # Load 4h data for entry timing and price action
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(range_ratio_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(close_1d_ma_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Range filter: only trade when daily range is expanded (volatile market)
        range_ratio_val = range_ratio_aligned[i]
        range_filter = range_ratio_val > 1.5
        
        # Volume filter: require above-average volume for confirmation
        vol_filter = vol_ratio_aligned[i] > 1.3
        
        # Trend filter: price above/below 50-day MA for directional bias
        close_1d_val = close_1d_aligned[i]
        close_1d_ma_50_val = close_1d_ma_50_aligned[i]
        uptrend = close_1d_val > close_1d_ma_50_val
        downtrend = close_1d_val < close_1d_ma_50_val
        
        if position == 0:
            # Long when: expanded range, high volume, and uptrend bias
            if range_filter and vol_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short when: expanded range, high volume, and downtrend bias
            elif range_filter and vol_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: when range contracts or trend reverses
            if not range_filter or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: when range contracts or trend reverses
            if not range_filter or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Range_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0