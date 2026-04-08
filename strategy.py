#!/usr/bin/env python3
# 4h_trend_follow_volume_v1
# Hypothesis: Trend following with 4h price above/below SMA50, confirmed by 1d SMA50 slope and volume surge.
# Long when: price > SMA50, 1d SMA50 slope > 0, volume > 1.5x 20-period average.
# Short when: price < SMA50, 1d SMA50 slope < 0, volume > 1.5x 20-period average.
# Exit when trend breaks (price crosses SMA50 opposite direction) or volume drops below average.
# Uses 3 conditions max to avoid overtrading. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trend_follow_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h SMA50 for trend
    sma_period = 50
    close_series = pd.Series(close)
    sma50 = close_series.rolling(window=sma_period, min_periods=sma_period).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for trend direction (SMA50 slope)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=sma_period, min_periods=sma_period).mean().values
    # Calculate slope: positive if current SMA > SMA 3 periods ago
    sma50_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(3, len(close_1d)):
        if not np.isnan(sma50_1d[i]) and not np.isnan(sma50_1d[i-3]):
            sma50_slope_1d[i] = sma50_1d[i] - sma50_1d[i-3]
    sma50_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(sma_period, vol_ma_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50[i]) or np.isnan(vol_ma[i]) or np.isnan(sma50_slope_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below SMA50 or volume drops below average
            if close[i] < sma50[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above SMA50 or volume drops below average
            if close[i] > sma50[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above SMA50, 1d SMA50 slope positive, volume surge
            if (close[i] > sma50[i] and 
                sma50_slope_1d_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below SMA50, 1d SMA50 slope negative, volume surge
            elif (close[i] < sma50[i] and 
                  sma50_slope_1d_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals