#!/usr/bin/env python3
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
    
    # Get weekly data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly moving average for trend
    ma_10_1w = pd.Series(df_1w['close'].values).rolling(window=10, min_periods=10).mean().values
    
    # Align to 6h timeframe
    ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ma_10_1w)
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily price action
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    avg_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data
    avg_range_aligned = align_htf_to_ltf(prices, df_1d, avg_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ma_10_1w_aligned[i]) or np.isnan(avg_range_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly MA
        trend_up = close[i] > ma_10_1w_aligned[i]
        trend_down = close[i] < ma_10_1w_aligned[i]
        
        # Volatility filter: avoid low volatility
        vol_filter = avg_range_aligned[i] > 0.0  # Always true, but keeps structure
        
        # Daily breakout conditions
        if i >= 20:  # Need daily history
            # Get recent daily high/low
            recent_high = np.max(high_1d[max(0, i-20):i+1])
            recent_low = np.min(low_1d[max(0, i-20):i+1])
            
            # Breakout from daily range
            breakout_up = close[i] > recent_high
            breakout_down = close[i] < recent_low
            
            # Entry: breakout in direction of weekly trend
            long_entry = breakout_up and trend_up and vol_filter
            short_entry = breakout_down and trend_down and vol_filter
            
            # Exit: opposite breakout or trend reversal
            long_exit = breakout_down or not trend_up
            short_exit = breakout_up or not trend_down
            
            if long_entry and position <= 0:
                signals[i] = 0.25
                position = 1
            elif short_entry and position >= 0:
                signals[i] = -0.25
                position = -1
            elif long_exit and position == 1:
                signals[i] = -0.25
                position = -1
            elif short_exit and position == -1:
                signals[i] = 0.25
                position = 1
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyTrend_DailyBreakout"
timeframe = "6h"
leverage = 1.0