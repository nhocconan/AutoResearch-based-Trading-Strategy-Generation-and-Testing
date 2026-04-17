# 12h_DailyPivot_Breakout_EMA34_Volume
# Strategy: Long when price breaks above daily R1 with volume and price above EMA34
# Short when price breaks below daily S1 with volume and price below EMA34
# Exit when price crosses back below/above daily pivot or EMA34
# Uses 12h timeframe to reduce trade frequency and capture multi-day trends
# EMA34 trend filter helps avoid whipsaws in choppy markets
# Volume confirmation ensures breakouts have institutional participation
# Target: 12-37 trades per year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align daily pivot levels to 12h timeframe
    daily_pivot_12h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_12h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_12h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need daily EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pivot_12h[i]) or 
            np.isnan(daily_r1_12h[i]) or 
            np.isnan(daily_s1_12h[i]) or 
            np.isnan(ema34_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema34_12h[i]
        price_below_ema = close[i] < ema34_12h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_12h[i]
        price_below_s1 = close[i] < daily_s1_12h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume and above daily EMA34
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume and below daily EMA34
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot OR below daily EMA34
            if (close[i] < daily_pivot_12h[i]) or (close[i] < ema34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot OR above daily EMA34
            if (close[i] > daily_pivot_12h[i]) or (close[i] > ema34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_Breakout_EMA34_Volume"
timeframe = "12h"
leverage = 1.0