#!/usr/bin/env python3
# daily_bollinger_breakout_volume_v1
# Hypothesis: Daily Bollinger Band breakout with volume confirmation and 1-week trend filter.
# Long when price breaks above upper BB(20,2) with volume > 1.5x avg and 1w SMA50 rising.
# Short when price breaks below lower BB(20,2) with volume > 1.5x avg and 1w SMA50 falling.
# Exit when price returns to middle BB or volume drops below average.
# Target: 15-25 trades/year to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_bollinger_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    std20 = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma20 + bb_std * std20
    lower_band = sma20 - bb_std * std20
    middle_band = sma20
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1w data for trend direction (SMA50 slope)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    # Calculate slope: positive if current SMA > SMA 3 periods ago
    sma50_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(3, len(close_1w)):
        if not np.isnan(sma50_1w[i]) and not np.isnan(sma50_1w[i-3]):
            sma50_slope_1w[i] = sma50_1w[i] - sma50_1w[i-3]
    sma50_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(bb_period, vol_ma_period, 50, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(sma50_slope_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below middle BB or volume drops below average
            if close[i] < middle_band[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above middle BB or volume drops below average
            if close[i] > middle_band[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper BB, 1w SMA50 slope positive, volume surge
            if (close[i] > upper_band[i] and 
                sma50_slope_1w_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower BB, 1w SMA50 slope negative, volume surge
            elif (close[i] < lower_band[i] and 
                  sma50_slope_1w_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals