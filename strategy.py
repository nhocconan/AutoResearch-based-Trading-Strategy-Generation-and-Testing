#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(13) and EMA(26) on daily close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA26
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema26_1d
    
    # Trend filter: EMA13 > EMA26 for uptrend, EMA13 < EMA26 for downtrend
    trend_up = ema13_1d > ema26_1d
    trend_down = ema13_1d < ema26_1d
    
    # Align to 6h timeframe (wait for daily close)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    trend_up_6h = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_6h = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(trend_up_6h[i]) or np.isnan(trend_down_6h[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Positive Bull Power + Uptrend + Volume confirmation
            if (bull_power_6h[i] > 0 and
                trend_up_6h[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Negative Bear Power + Downtrend + Volume confirmation
            elif (bear_power_6h[i] < 0 and
                  trend_down_6h[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power turns negative or trend turns down
            if (bear_power_6h[i] < 0 or
                not trend_up_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power turns positive or trend turns up
            if (bull_power_6h[i] > 0 or
                not trend_down_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals