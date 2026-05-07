#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1d
    # Using previous day's high, low, close
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (phigh + plow + pclose) / 3.0
    range_val = phigh - plow
    
    # Camarilla levels: R3 = pivot + 1.1 * range/2, S3 = pivot - 1.1 * range/2
    r3 = pivot + 1.1 * range_val / 2.0
    s3 = pivot - 1.1 * range_val / 2.0
    
    # Align to 12h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2.0x 24-period average (for 12h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~1 day for 12h to reduce trades
    
    start_idx = max(100, 24, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
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
            # Long: Price breaks above R3 with volume in uptrend
            if (close[i] > r3_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S3 with volume in downtrend
            elif (close[i] < s3_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below S3 or trend changes
            if close[i] < s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above R3 or trend changes
            if close[i] > r3_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation on 12h timeframe.
# Long when price breaks above Camarilla R3 in uptrend with volume confirmation.
# Short when price breaks below Camarilla S3 in downtrend with volume confirmation.
# Uses 12h timeframe to balance trade frequency and capture meaningful trends.
# Target: 50-150 total trades over 4 years (12-37/year) as per experiment guidelines.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Based on top-performing pattern from DB: Camarilla pivot + volume + trend filter.