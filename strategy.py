#!/usr/bin/env python3
# 12h_camarilla_pivot_daily_trend_volume_v3
# Hypothesis: Uses daily Camarilla pivot levels with 1w trend filter and volume confirmation for 12h entries.
# Camarilla levels provide high-probability reversal points; trend filter ensures directional bias.
# Volume confirmation reduces false breakouts. Designed for low-frequency, high-edge trades in all market regimes.
# Target: 15-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_daily_trend_volume_v3"
timeframe = "12h"
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
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # Using previous day's range to avoid look-ahead
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    cam_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    cam_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 12h timeframe
    cam_h4_aligned = align_htf_to_ltf(prices, df_1d, cam_h4)
    cam_l4_aligned = align_htf_to_ltf(prices, df_1d, cam_l4)
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are warmed up
    
    for i in range(start_idx, n):
        if (np.isnan(cam_h4_aligned[i]) or np.isnan(cam_l4_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filters: daily and weekly
        daily_uptrend = close[i] > ema200_1d_aligned[i]
        daily_downtrend = close[i] < ema200_1d_aligned[i]
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Camarilla L4 level or weekly trend breaks
            if close[i] <= cam_l4_aligned[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla H4 level or weekly trend breaks
            if close[i] >= cam_h4_aligned[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long setup: price crosses above Camarilla H4 with daily/weekly uptrend
                if (daily_uptrend and weekly_uptrend and 
                    close[i] > cam_h4_aligned[i] and close[i-1] <= cam_h4_aligned[i-1]):
                    position = 1
                    signals[i] = 0.25
                # Short setup: price crosses below Camarilla L4 with daily/weekly downtrend
                elif (daily_downtrend and weekly_downtrend and 
                      close[i] < cam_l4_aligned[i] and close[i-1] >= cam_l4_aligned[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals