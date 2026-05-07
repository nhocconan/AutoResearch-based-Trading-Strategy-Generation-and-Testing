#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla pivot levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, R3, S1, S3
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Close + (Range * 1.0833)
    # R3 = Close + (Range * 1.2500)
    # S1 = Close - (Range * 1.0833)
    # S3 = Close - (Range * 1.2500)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.0833)
    r3_1d = close_1d + (range_1d * 1.2500)
    s1_1d = close_1d - (range_1d * 1.0833)
    s3_1d = close_1d - (range_1d * 1.2500)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: current volume > 1.8x 24-period average (for 12h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~1 day for 12h to reduce trades
    
    start_idx = max(100, 24, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
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
        
        # Determine 1d trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above R1 with volume in uptrend
            if (close[i] > r1_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S1 with volume in downtrend
            elif (close[i] < s1_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below S1 or trend changes
            if close[i] < s1_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above R1 or trend changes
            if close[i] > r1_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation on 12h timeframe.
# Long when price breaks above Camarilla R1 in uptrend with volume confirmation.
# Short when price breaks below Camarilla S1 in downtrend with volume confirmation.
# Uses 12h timeframe to balance trade frequency and capture meaningful trends.
# Target: 50-150 total trades over 4 years (12-37/year) as per experiment guidelines.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Based on top-performing pattern from DB: Camarilla breakout + volume + trend filter.