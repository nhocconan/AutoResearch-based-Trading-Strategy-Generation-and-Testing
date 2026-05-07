#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 from previous day's OHLC
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    for i in range(1, n):
        prev_high = df_1d['high'].values[i-1]
        prev_low = df_1d['low'].values[i-1]
        prev_close = df_1d['close'].values[i-1]
        camarilla_R3[i] = prev_close + (prev_high - prev_low) * 1.1 / 4
        camarilla_S3[i] = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~1 day for 12h to reduce trades
    
    start_idx = max(100, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        trend_up = close > ema_50_1w_aligned[i]
        trend_down = close < ema_50_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R3 with volume in uptrend
            if (close[i] > camarilla_R3[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume in downtrend
            elif (close[i] < camarilla_S3[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below Camarilla S3 or trend changes
            if close[i] < camarilla_S3[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above Camarilla R3 or trend changes
            if close[i] > camarilla_R3[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with weekly EMA50 trend filter and volume confirmation on 12h timeframe.
# Long when price breaks above Camarilla R3 in uptrend with volume confirmation.
# Short when price breaks below Camarilla S3 in downtrend with volume confirmation.
# Uses 12h timeframe to balance trade frequency and capture meaningful trends.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Camarilla levels provide precise support/resistance based on prior day's range.
# Volume confirmation reduces false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) as per experiment guidelines.