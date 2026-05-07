#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # 1d trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Camarilla levels from previous 1d
    close_prev_1d = df_1d['close'].values
    high_prev_1d = df_1d['high'].values
    low_prev_1d = df_1d['low'].values
    range_prev_1d = high_prev_1d - low_prev_1d
    # R3 and S3 levels
    r3 = close_prev_1d + range_prev_1d * 1.1 / 4
    s3 = close_prev_1d - range_prev_1d * 1.1 / 4
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume surge: current volume > 2.5x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_surge = volume > (2.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~24 hours (2*12h) to reduce trade frequency
    
    start_idx = max(34, 24)  # Ensure enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above R3 with volume surge in 1d uptrend
            if (close[i] > r3_aligned[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.30
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S3 with volume surge in 1d downtrend
            elif (close[i] < s3_aligned[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.30
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below S3 or 1d trend changes to down
            if close[i] < s3_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price breaks above R3 or 1d trend changes to up
            if close[i] > r3_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Breakout at 1d Camarilla R3/S3 with volume surge and 1d trend filter on 12h timeframe.
# Uses tighter levels (R3/S3) and higher volume threshold (2.5x) to reduce trade frequency.
# Trend filter uses 1d EMA34 for stronger trend confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).