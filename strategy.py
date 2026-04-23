#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R combined with 1-day Bollinger Band squeeze breakout and volume confirmation.
Long when Williams %R crosses above -20 (bullish), price breaks above upper Bollinger Band (20,2), and volume > 1.5x average.
Short when Williams %R crosses below -80 (bearish), price breaks below lower Bollinger Band (20,2), and volume > 1.5x average.
Exit when Williams %R crosses back through -50 or Bollinger Band width expands beyond 1.5x average width.
Williams %R identifies momentum extremes, Bollinger Bands identify volatility breakouts, volume confirms strength.
Designed for low trade frequency (~15-25/year) to capture strong momentum breaks while minimizing false signals.
Works in both bull and bear markets by requiring volatility expansion (Bollinger Band breakout) with momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 12-hour data for Williams %R - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20,2)
    close_1d = df_1d['close'].values
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Bollinger Band width (for exit condition)
    bb_width = upper_bb - lower_bb
    bb_width_avg = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12-hour Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align HTF indicators to lower timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_width_avg_aligned = align_htf_to_ltf(prices, df_1d, bb_width_avg)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(bb_width_avg_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r_aligned[i]
        williams_r_prev = williams_r_aligned[i-1]
        close_price = close[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        bb_width_avg_val = bb_width_avg_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -20, price breaks above upper BB, volume confirmation
            if (williams_r_val > -20 and williams_r_prev <= -20 and
                close_price > upper_bb_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80, price breaks below lower BB, volume confirmation
            elif (williams_r_val < -80 and williams_r_prev >= -80 and
                  close_price < lower_bb_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -50 OR Bollinger Band width expands significantly
                if (williams_r_val < -50 and williams_r_prev >= -50) or bb_width_avg_val > 1.5 * bb_width_avg[i-20] if i >= 20 else False:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses above -50 OR Bollinger Band width expands significantly
                if (williams_r_val > -50 and williams_r_prev <= -50) or bb_width_avg_val > 1.5 * bb_width_avg[i-20] if i >= 20 else False:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1dBB_Breakout_Volume"
timeframe = "12h"
leverage = 1.0