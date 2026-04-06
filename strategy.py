#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian high + weekly pivot in uptrend + volume > 1.5x average
# Short when price breaks below Donchian low + weekly pivot in downtrend + volume > 1.5x average
# Exit when price crosses Donchian midpoint
# Uses weekly pivot (calculated from prior week) to filter direction, reducing whipsaw in ranging markets
# Targets 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_donchian_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 6h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Weekly pivot points (calculated from prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivot from previous week's data (shifted by 1 to avoid lookahead)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Classic pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Support/resistance levels
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Determine weekly trend: price above/below weekly pivot
    weekly_trend_up = weekly_pivot > weekly_s1  # Uptrend if pivot above S1
    weekly_trend_down = weekly_pivot < weekly_r1  # Downtrend if pivot below R1
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or
            np.isnan(weekly_trend_down_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midpoint
        if position == 1:  # long position
            if close[i] <= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with weekly trend and volume confirmation
            # Bullish breakout: price above Donchian high + weekly uptrend + volume
            if (close[i] > donch_high[i] and 
                weekly_trend_up_aligned[i] > 0.5 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + weekly downtrend + volume
            elif (close[i] < donch_low[i] and 
                  weekly_trend_down_aligned[i] > 0.5 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals