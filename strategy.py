#!/usr/bin/env python3
"""
12h_WPivot_R1_S1_Breakout_VolumeFilter_v3
Strategy: Weekly pivot point breakout with volume confirmation and trend filter.
Long: Price breaks above weekly S1 + price above 1d EMA34 + volume > 1.5x average
Short: Price breaks below weekly S1 + price below 1d EMA34 + volume > 1.5x average
Exit: Price moves back inside weekly pivot range
Position size: 0.25
Designed to capture breakouts from weekly pivot levels with trend alignment and volume confirmation.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        # Calculate weekly pivot and S1 from previous week
        if i >= 14:  # Need at least 14 12h bars (1 week) to get previous week
            # Get weekly data for S1 calculation
            df_1w = get_htf_data(prices, '1w')
            
            # Find previous week's index in 1w data
            current_time = prices['open_time'].iloc[i]
            prev_week = current_time - pd.Timedelta(days=7)
            
            # Get previous week's OHLC from 1w data
            week_mask = df_1w['open_time'].dt.date == prev_week.date()
            if week_mask.any():
                prev_week_data = df_1w[week_mask].iloc[0]
                prev_high = prev_week_data['high']
                prev_low = prev_week_data['low']
                prev_close = prev_week_data['close']
                
                # Calculate weekly pivot and S1
                pivot = (prev_high + prev_low + prev_close) / 3
                range_val = prev_high - prev_low
                if range_val > 0:
                    s1 = pivot - (range_val * 1.1 / 12)
                    
                    # Entry conditions: price breaks above/below S1
                    price_above_s1 = close[i] > s1
                    price_below_s1 = close[i] < s1
                    
                    if position == 0:
                        # Long: breaks above S1 + price above EMA + volume filter
                        if price_above_s1 and price_above_ema and volume_filter:
                            signals[i] = 0.25
                            position = 1
                        # Short: breaks below S1 + price below EMA + volume filter
                        elif price_below_s1 and price_below_ema and volume_filter:
                            signals[i] = -0.25
                            position = -1
                    
                    elif position == 1:
                        # Exit long: price moves back inside pivot range
                        if close[i] < pivot:
                            signals[i] = 0.0
                            position = 0
                        else:
                            signals[i] = 0.25
                    
                    elif position == -1:
                        # Exit short: price moves back inside pivot range
                        if close[i] > pivot:
                            signals[i] = 0.0
                            position = 0
                        else:
                            signals[i] = -0.25
                else:
                    signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WPivot_R1_S1_Breakout_VolumeFilter_v3"
timeframe = "12h"
leverage = 1.0