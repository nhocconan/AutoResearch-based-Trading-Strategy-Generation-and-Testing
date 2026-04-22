#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Daily Fibonacci Extension Breakout with Weekly Trend Filter and Volume Spike
    # Uses 1d Fibonacci extensions (1.272, 1.618) from prior week's range as key levels
    # Weekly EMA200 filters trend direction, volume surge confirms breakout strength
    # Fibonacci extensions provide strong support/resistance in both bull and bear markets
    # Weekly trend filter ensures we trade with the higher timeframe momentum
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 trend filter
    ema_1w_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Load daily data for Fibonacci extensions (based on prior week OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high/low for Fibonacci calculations
    # We need to aggregate daily data to weekly to get proper weekly range
    # But since we already have weekly data from df_1w, we can use that
    # However, we need to align weekly high/low to daily timeframe
    
    # Extract weekly high and low from weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align weekly high/low to daily timeframe (each weekly value applies to all days of that week)
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Calculate Fibonacci extension levels from weekly range
    weekly_range = high_1w_aligned - low_1w_aligned
    fib_extension_1272 = low_1w_aligned + 1.272 * weekly_range  # 127.2% extension
    fib_extension_1618 = low_1w_aligned + 1.618 * weekly_range  # 161.8% extension
    
    # For short signals, we use extensions from the top downward
    fib_extension_1272_short = high_1w_aligned - 1.272 * weekly_range
    fib_extension_1618_short = high_1w_aligned - 1.618 * weekly_range
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_200_aligned[i]) or np.isnan(fib_extension_1272[i]) or 
            np.isnan(fib_extension_1618[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 1.272 extension with volume spike and weekly uptrend
            if close[i] > fib_extension_1272[i] and vol_spike[i] and close[i] > ema_1w_200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below 1.272 extension (from top) with volume spike and weekly downtrend
            elif close[i] < fib_extension_1272_short[i] and vol_spike[i] and close[i] < ema_1w_200_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to weekly VWAP or opposite extension level
            # Calculate weekly VWAP approximation using weekly data
            # VWAP = sum(price * volume) / sum(volume) for the week
            # We'll use a simplified approach: midpoint of weekly range
            weekly_midpoint = (high_1w_aligned + low_1w_aligned) / 2.0
            
            if position == 1:
                if close[i] < weekly_midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > weekly_midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Fibonacci_Extension_Breakout_WeeklyEMA200_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0