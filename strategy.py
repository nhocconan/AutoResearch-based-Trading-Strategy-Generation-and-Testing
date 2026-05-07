#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1
Hypothesis: On 12h timeframe, use Camarilla R3/S3 levels from prior day for breakout signals,
filtered by daily EMA trend and volume spikes. Long when price breaks above R3 with daily
uptrend and volume spike. Short when price breaks below S3 with daily downtrend and volume spike.
Camarilla levels provide strong intraday support/resistance, effective in both trending and
ranging markets. 12h timeframe reduces trade frequency to avoid fee drag while capturing
significant moves.
"""
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import ceil
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero and handle first bar
    price_range = prev_high - prev_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    r3 = prev_close + price_range * 1.1 / 2
    s3 = prev_close - price_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    
    start_idx = 1  # Need at least one previous day
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades to reduce frequency (12h timeframe)
            if bars_since_exit < 6:
                continue
                
            # Long: price breaks above R3 + daily uptrend + volume filter
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S3 + daily downtrend + volume filter
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or EMA crossover
            if position == 1:
                if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = -0.25
    
    return signals