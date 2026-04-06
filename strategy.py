#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action with 1d trend filter and volume confirmation.
# Long when price breaks above 4h high of previous day during bullish 1d with volume > 1.5x 20-period average.
# Short when price breaks below 4h low of previous day during bearish 1d with volume confirmation.
# Uses previous day's high/low as dynamic support/resistance. Target: 100-200 total trades over 4 years.

name = "4h_prevday_breakout_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prates)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Previous day high/low from 1d data
    df_1d = get_htf_data(prices, '1d')
    prev_day_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_day_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_day_open = df_1d['open'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Daily trend: bullish if previous day close > open
    daily_bullish = prev_day_close > prev_day_open
    daily_bearish = prev_day_close < prev_day_open
    
    # Align 1d data to 4h
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 1d data not available
        if np.isnan(prev_day_high_aligned[i]) or np.isnan(prev_day_low_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below previous day low or daily turn bearish
            if (low[i] <= prev_day_low_aligned[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above previous day high or daily turn bullish
            if (high[i] >= prev_day_high_aligned[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: break above previous day high during bullish day
                if (high[i] > prev_day_high_aligned[i] and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below previous day low during bearish day
                elif (low[i] < prev_day_low_aligned[i] and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals