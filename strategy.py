#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    R3 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    
    for i in range(n):
        # Find the most recent completed 1d bar
        # Use the 1d data index to get proper alignment
        idx_1d = np.searchsorted(df_1d.index, prices.index[i], side='right') - 1
        if idx_1d < 0:
            continue
        if idx_1d >= len(df_1d):
            idx_1d = len(df_1d) - 1
        
        H = high_1d[idx_1d]
        L = low_1d[idx_1d]
        C = close_1d[idx_1d]
        R = H - L
        
        camarilla_high[i] = C + (R * 1.1 / 2)  # R4 level
        camarilla_low[i] = C - (R * 1.1 / 2)   # S4 level
        R3[i] = C + (R * 1.1 / 4)              # R3 level
        S3[i] = C - (R * 1.1 / 4)              # S3 level
    
    # Volume filter: current volume > 1.5x 20-period average (for 4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~6 hours for 4h to reduce trades
    
    start_idx = max(100, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_high[i]) or 
            np.isnan(camarilla_low[i]) or 
            np.isnan(R3[i]) or 
            np.isnan(S3[i]) or 
            np.isnan(vol_ma_20[i])):
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
            # Long: Price breaks above R3 with volume in uptrend
            if (close[i] > R3[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S3 with volume in downtrend
            elif (close[i] < S3[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below S3 or trend changes
            if close[i] < S3[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above R3 or trend changes
            if close[i] > R3[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation on 4h timeframe.
# Long when price breaks above R3 level in uptrend with volume confirmation.
# Short when price breaks below S3 level in downtrend with volume confirmation.
# Uses 4h timeframe to balance trade frequency and capture meaningful trends.
# Target: 75-200 total trades over 4 years (19-50/year) as per experiment guidelines.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Based on top-performing pattern from DB: Camarilla breakout + volume + trend filter.