#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v2"
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
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels (R3, R2, R1, PP, S1, S2, S3) from 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for pivot levels
    R3 = np.full(n, np.nan)
    R2 = np.full(n, np.nan)
    R1 = np.full(n, np.nan)
    PP = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    S2 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    
    # Calculate pivot levels for each completed 1d bar
    for i in range(len(df_1d)):
        H = high_1d[i]
        L = low_1d[i]
        C = close_1d[i]
        range_val = H - L
        if range_val <= 0:
            continue
        pp = (H + L + C) / 3
        r3 = C + range_val * 1.1 / 4
        r2 = C + range_val * 1.1 / 2
        r1 = C + range_val * 1.1 / 4
        s1 = C - range_val * 1.1 / 4
        s2 = C - range_val * 1.1 / 2
        s3 = C - range_val * 1.1 / 4
        # Align to LTF: each 1d bar affects 6*4h bars (24h/4h)
        start_idx = i * 6
        end_idx = min(start_idx + 6, n)
        if start_idx < n:
            R3[start_idx:end_idx] = r3
            R2[start_idx:end_idx] = r2
            R1[start_idx:end_idx] = r1
            PP[start_idx:end_idx] = pp
            S1[start_idx:end_idx] = s1
            S2[start_idx:end_idx] = s2
            S3[start_idx:end_idx] = s3
    
    # Volume filter: current volume > 1.3x 20-period average (for 4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~8 hours for 4h to reduce trades
    
    start_idx = max(100, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3[i]) or np.isnan(R2[i]) or np.isnan(R1[i]) or
            np.isnan(PP[i]) or np.isnan(S1[i]) or np.isnan(S2[i]) or np.isnan(S3[i]) or
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
            # Long: Price crosses above R3 with volume in uptrend
            if (close[i] > R3[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price crosses below S3 with volume in downtrend
            elif (close[i] < S3[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price crosses below R1 or trend changes
            if close[i] < R1[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above S1 or trend changes
            if close[i] > S1[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation on 4h timeframe.
# Long when price breaks above Camarilla R3 level in uptrend with volume confirmation.
# Short when price breaks below Camarilla S3 level in downtrend with volume confirmation.
# Uses 4h timeframe to balance trade frequency and capture meaningful trends.
# Target: 75-200 total trades over 4 years (19-50/year) as per experiment guidelines.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Based on top-performing pattern from DB: Camarilla pivot breakout + volume + trend filter.