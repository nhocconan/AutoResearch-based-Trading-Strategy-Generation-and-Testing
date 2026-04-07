#!/usr/bin/env python3
"""
1d_donchian_20_breakout_1w_trend_volume_v1
Hypothesis: On 1d timeframe, enter long when price breaks above 20-day Donchian high with 1w uptrend and volume confirmation.
Enter short when price breaks below 20-day Donchian low with 1w downtrend and volume confirmation.
Uses 1w EMA20 for trend filter and 20-day average volume for confirmation.
Targets 7-25 trades/year to minimize fee drag while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 on weekly close
    weekly_close = df_1w['close'].values
    weekly_close_s = pd.Series(weekly_close)
    ema20_1w = weekly_close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align to 1d timeframe (shifted by 1 week to avoid look-ahead)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian20 warmup
        # Skip if required data not available
        if (np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate 20-day Donchian channels
        # Lookback 20 periods including current
        start_idx = max(0, i - 19)
        highest_high = np.max(high[start_idx:i+1])
        lowest_low = np.min(low[start_idx:i+1])
        
        # Volume confirmation: current volume > 1.3x 20-day average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter from 1w: up if close > EMA20, down if close < EMA20
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price breaks below 10-day Donchian low (shorter lookback for faster exit)
            if i >= 9:
                lowest_low_10 = np.min(low[i-9:i+1])
                if close[i] < lowest_low_10:
                    exit_long = True
            # Exit when trend reverses
            elif not trend_up:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price breaks above 10-day Donchian high
            if i >= 9:
                highest_high_10 = np.max(high[i-9:i+1])
                if close[i] > highest_high_10:
                    exit_short = True
            # Exit when trend reverses
            elif not trend_down:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-day Donchian high, 1w trend up, volume confirmation
            long_entry = (close[i] > highest_high) and trend_up and vol_confirm
            
            # Short entry: price breaks below 20-day Donchian low, 1w trend down, volume confirmation
            short_entry = (close[i] < lowest_low) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals