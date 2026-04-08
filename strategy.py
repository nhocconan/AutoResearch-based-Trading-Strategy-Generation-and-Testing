#!/usr/bin/env python3
# 6h_ema_pullback_weekly_trend_v1
# Hypothesis: 6s pullback to EMA21 with weekly trend filter (EMA50 slope) and volume confirmation.
# Long when price > EMA21, weekly EMA50 slope > 0, volume > 1.5x average.
# Short when price < EMA21, weekly EMA50 slope < 0, volume > 1.5x average.
# Designed for 6h timeframe to capture swings in both bull and bear markets with controlled trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_pullback_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA21 for pullback entries
    ema_period = 21
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for trend direction (EMA50 slope)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate slope: positive if current EMA > EMA 3 periods ago
    ema50_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(3, len(close_1w)):
        if not np.isnan(ema50_1w[i]) and not np.isnan(ema50_1w[i-3]):
            ema50_slope_1w[i] = ema50_1w[i] - ema50_1w[i-3]
    ema50_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(ema_period, vol_ma_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50_slope_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below EMA21 or volume drops below average
            if close[i] < ema21[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above EMA21 or volume drops below average
            if close[i] > ema21[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above EMA21, weekly EMA50 slope positive, volume surge
            if (close[i] > ema21[i] and 
                ema50_slope_1w_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below EMA21, weekly EMA50 slope negative, volume surge
            elif (close[i] < ema21[i] and 
                  ema50_slope_1w_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals