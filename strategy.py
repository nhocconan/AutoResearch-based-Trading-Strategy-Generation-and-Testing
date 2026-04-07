#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_ema_volume_v1
Hypothesis: On 12-hour timeframe, use weekly (1w) daily price extremes for weekly context and daily (1d) EMA for trend, with volume confirmation. Enter long when price breaks above weekly high with rising daily EMA and volume > 1.5x average; enter short when price breaks below weekly low with falling daily EMA and volume > 1.5x average. Exit on opposite weekly extreme or trend reversal. Weekly context provides structural bias, daily EMA provides intermediate trend, volume confirms institutional participation. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for structural context
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly high/low for structural context
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly high/low to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA(50) to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 12-period average on 12h timeframe (equivalent to ~6 days)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(12, 50), n):
        # Skip if data not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches weekly low (stop) or weekly high (take profit) or trend reversal
            if low[i] <= weekly_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif high[i] >= weekly_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:  # Trend reversal
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches weekly high (stop) or weekly low (take profit) or trend reversal
            if high[i] >= weekly_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif low[i] <= weekly_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:  # Trend reversal
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout entry: break above weekly high with rising EMA
                if high[i] >= weekly_high_aligned[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown entry: break below weekly low with falling EMA
                elif low[i] <= weekly_low_aligned[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals