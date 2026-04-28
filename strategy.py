#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly Donchian breakout with daily volume confirmation and 1d EMA50 trend filter.
# Enter long when price breaks above weekly Donchian(20) upper band with volume > 2.0x daily average and close > 1d EMA50.
# Enter short when price breaks below weekly Donchian(20) lower band with volume > 2.0x daily average and close < 1d EMA50.
# Exit when price touches the opposite Donchian band or returns to the weekly midpoint.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 50-150 total trades over 4 years.
# Weekly structure provides stability, daily volume confirms institutional interest, EMA50 filters counter-trend noise.
# Works in bull markets (breakouts continue with trend) and bear markets (breakdowns continue with trend).

name = "12h_Donchian20_1dVolume_EMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel calculation (HTF structure)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian upper and lower bands (20-period)
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align weekly Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
    
    # Get daily data for volume confirmation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily volume confirmation: >2.0x 20-bar average volume
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = volume_1d / volume_ma_20_1d
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: >2.0x daily average volume
        vol_confirm = volume_ratio_aligned[i] > 2.0
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > upper_aligned[i]
        short_breakout = close[i] < lower_aligned[i]
        
        # Exit conditions: touch opposite band or return to midpoint
        long_exit = close[i] < lower_aligned[i]  # Touch lower band (opposite)
        short_exit = close[i] > upper_aligned[i]  # Touch upper band (opposite)
        midpoint_exit = (
            (position == 1 and close[i] < midpoint_aligned[i]) or  # Long exits at midpoint
            (position == -1 and close[i] > midpoint_aligned[i])    # Short exits at midpoint
        )
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and (long_exit or midpoint_exit)) or \
             (position == -1 and (short_exit or midpoint_exit)):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals