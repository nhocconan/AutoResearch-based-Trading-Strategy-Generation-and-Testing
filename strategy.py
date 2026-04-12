#!/usr/bin/env python3
"""
1d_1w_Aroon_Breakout_Trend_v1
Hypothesis: Uses weekly Aroon trend strength to identify strong trends, then enters on daily Aroon crossovers in the trend direction. 
Aroon > 70 indicates strong trend, reducing false signals. Works in both bull and bear markets by following the weekly trend.
Targets 8-20 trades/year per symbol with high-probability trend-following entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Aroon_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0

def aroon_up(down_period, n):
    """Calculate Aroon Up: ((n - periods since highest high) / n) * 100"""
    return ((n - np.argmax(down_period[::-1])) / n) * 100 if len(down_period) > 0 else np.nan

def aroon_down(up_period, n):
    """Calculate Aroon Down: ((n - periods since lowest low) / n) * 100"""
    return ((n - np.argmin(up_period[::-1])) / n) * 100 if len(up_period) > 0 else np.nan

def calculate_aroon(high, low, period=25):
    """Calculate Aroon Up and Down arrays"""
    n = len(high)
    aroon_up_arr = np.full(n, np.nan)
    aroon_down_arr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window_high = high[i - period + 1:i + 1]
        window_low = low[i - period + 1:i + 1]
        aroon_up_arr[i] = aroon_up(window_high, period)
        aroon_down_arr[i] = aroon_down(window_low, period)
    
    return aroon_up_arr, aroon_down_arr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Aroon trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Calculate weekly Aroon for trend strength (25 periods)
    aroon_up_1w, aroon_down_1w = calculate_aroon(df_1w['high'].values, df_1w['low'].values, 25)
    strong_uptrend_1w = aroon_up_1w > 70  # Strong uptrend when Aroon Up > 70
    strong_downtrend_1w = aroon_down_1w > 70  # Strong downtrend when Aroon Down > 70
    
    # Align weekly Aroon signals to daily timeframe
    strong_uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, strong_uptrend_1w.astype(float))
    strong_downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, strong_downtrend_1w.astype(float))
    
    # Calculate daily Aroon for entry signals (25 periods)
    aroon_up_daily, aroon_down_daily = calculate_aroon(high, low, 25)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(25, n):
        # Skip if any required data is invalid
        if (np.isnan(aroon_up_daily[i]) or np.isnan(aroon_down_daily[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(strong_uptrend_1w_aligned[i]) or 
            np.isnan(strong_downtrend_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Daily Aroon crossover signals
        aroon_cross_up = (aroon_up_daily[i] > aroon_down_daily[i]) and (aroon_up_daily[i-1] <= aroon_down_daily[i-1])
        aroon_cross_down = (aroon_down_daily[i] > aroon_up_daily[i]) and (aroon_down_daily[i-1] <= aroon_up_daily[i-1])
        
        # Entry conditions: only trade in direction of weekly trend
        long_entry = aroon_cross_up and volume_filter and strong_uptrend_1w_aligned[i] > 0.5
        short_entry = aroon_cross_down and volume_filter and strong_downtrend_1w_aligned[i] > 0.5
        
        # Exit conditions: opposite Aroon crossover or loss of weekly trend strength
        long_exit = aroon_cross_down or strong_uptrend_1w_aligned[i] <= 0.5
        short_exit = aroon_cross_up or strong_downtrend_1w_aligned[i] <= 0.5
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals