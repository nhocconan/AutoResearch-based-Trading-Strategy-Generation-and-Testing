#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Filter_VolumeSpike
Hypothesis: Use 4h Camarilla pivot (R1/S1) breakout for signal direction on 1h timeframe.
Filter with 4h EMA50 trend and volume spike confirmation. 1h timeframe provides entry timing
precision while 4h trend reduces whipsaw. Target 15-37 trades/year to stay within fee limits.
Works in bull/bear markets by following 4h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate 4h Camarilla pivot levels using previous day's OHLC
        # Convert 1h index to 4h bar index
        hour_4h_idx = i // 4  # 4 1h bars per 4h bar
        # Convert to daily index: 6 4h bars per day
        day_idx = hour_4h_idx // 6
        if day_idx < 1:
            signals[i] = 0.0
            continue
            
        prev_day_idx = day_idx - 1
        if prev_day_idx >= len(df_4h):
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC from 4h data (resampled daily)
        # We need to get the OHLC for the previous day from 4h bars
        start_bar = prev_day_idx * 6
        end_bar = start_bar + 6
        if end_bar > len(df_4h):
            signals[i] = 0.0
            continue
            
        # Get the 4h bars for previous day
        day_high = np.max(df_4h['high'].iloc[start_bar:end_bar])
        day_low = np.min(df_4h['low'].iloc[start_bar:end_bar])
        day_close = df_4h['close'].iloc[end_bar - 1]  # Last 4h bar of previous day
        
        # Camarilla levels
        range_val = day_high - day_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        r1 = day_close + (range_val * 1.1 / 12)
        s1 = day_close - (range_val * 1.1 / 12)
        
        # Trend direction from 4h EMA50
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation: >2.0x 20-period MA
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Breakout conditions
        long_breakout = close[i] > r1
        short_breakout = close[i] < s1
        
        # Entry logic
        long_entry = vol_confirm and trend_up and long_breakout
        short_entry = vol_confirm and trend_down and short_breakout
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = (close[i] < s1) or (not trend_up)
        short_exit = (close[i] > r1) or (not trend_down)
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Filter_VolumeSpike"
timeframe = "1h"
leverage = 1.0