#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter
Strategy: Camarilla R1/S1 breakout on 12h with 1d trend filter and volume confirmation.
Long: Close breaks above R1 + 1d EMA34 > EMA144 + volume > 1.5x average
Short: Close breaks below S1 + 1d EMA34 < EMA144 + volume > 1.5x average
Exit: Close returns to CAM (pivot) or trend reverses
Position size: 0.25
Designed to capture institutional breakouts with trend alignment and volume confirmation.
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
    
    # Calculate 1d EMA34 and EMA144 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema144_1d = close_series_1d.ewm(span=144, adjust=False, min_periods=144).mean().values
    
    # Align 1d EMAs to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema144_1d_aligned = align_htf_to_ltf(prices, df_1d, ema144_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(144, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema144_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: 1d EMA34 > EMA144 for long, < for short
        ema34_gt_ema144 = ema34_1d_aligned[i] > ema144_1d_aligned[i]
        ema34_lt_ema144 = ema34_1d_aligned[i] < ema144_1d_aligned[i]
        
        # Calculate Camarilla levels from previous day's OHLC
        # Need previous day's data - use daily OHLC from 1d timeframe
        if i >= 2:  # Need at least 2 12h bars (1 day) to get previous day
            # Get daily data for Camarilla calculation
            
            # Find previous day's index in 1d data
            # Current 12h bar timestamp
            current_time = prices['open_time'].iloc[i]
            # Previous day's date
            prev_day = current_time - pd.Timedelta(days=1)
            
            # Get previous day's OHLC from 1d data
            day_mask = df_1d['open_time'].dt.date == prev_day.date()
            if day_mask.any():
                prev_day_data = df_1d[day_mask].iloc[0]
                prev_high = prev_day_data['high']
                prev_low = prev_day_data['low']
                prev_close = prev_day_data['close']
                
                # Calculate Camarilla levels
                range_val = prev_high - prev_low
                if range_val > 0:
                    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
                    camarilla_r1 = camarilla_pivot + (range_val * 1.1 / 12)
                    camarilla_s1 = camarilla_pivot - (range_val * 1.1 / 12)
                    
                    # Entry conditions
                    if position == 0:
                        # Long: Close breaks above R1 + trend up + volume
                        if (close[i] > camarilla_r1 and ema34_gt_ema144 and volume_filter):
                            signals[i] = 0.25
                            position = 1
                        # Short: Close breaks below S1 + trend down + volume
                        elif (close[i] < camarilla_s1 and ema34_lt_ema144 and volume_filter):
                            signals[i] = -0.25
                            position = -1
                    
                    elif position == 1:
                        # Exit long: Close returns to pivot or trend reverses
                        if close[i] < camarilla_pivot or not ema34_gt_ema144:
                            signals[i] = 0.0
                            position = 0
                        else:
                            signals[i] = 0.25
                    
                    elif position == -1:
                        # Exit short: Close returns to pivot or trend reverses
                        if close[i] > camarilla_pivot or not ema34_lt_ema144:
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

name = "12h_1d_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0