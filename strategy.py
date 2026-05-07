#!/usr/bin/env python3

name = "4h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # Get 1w data for trend filter (primary) and 1d for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d_prev = high_1d[:-1]
    low_1d_prev = low_1d[:-1]
    close_1d_prev = close_1d[:-1]
    high_1d_prev = np.concatenate([[np.nan], high_1d_prev])
    low_1d_prev = np.concatenate([[np.nan], low_1d_prev])
    close_1d_prev = np.concatenate([[np.nan], close_1d_prev])
    
    # R3 = H3 = close + 1.1*(high-low)/4
    # S3 = L3 = close - 1.1*(high-low)/4
    high_low = high_1d_prev - low_1d_prev
    r3 = close_1d_prev + 1.1 * high_low / 4
    s3 = close_1d_prev - 1.1 * high_low / 4
    
    # Align indicators to 4h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume filter: current volume > 1.8x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # ~2 days for 4h to reduce trades
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1w trend direction
        trend_1w_up = close_1d_aligned[i] > ema_50_1w_aligned[i]
        trend_1w_down = close_1d_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above R3 with volume in 1w uptrend and sufficient volatility
            if (close[i] > r3_aligned[i] and 
                trend_1w_up and 
                vol_filter[i] and
                atr_14_1d_aligned[i] > 0):  # Ensure volatility is present
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: break below S3 with volume in 1w downtrend and sufficient volatility
            elif (close[i] < s3_aligned[i] and 
                  trend_1w_down and 
                  vol_filter[i] and
                  atr_14_1d_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: close back below S3 or trend change
            if (close[i] < s3_aligned[i]) or not trend_1w_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close back above R3 or trend change
            if (close[i] > r3_aligned[i]) or not trend_1w_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 1w EMA50 trend filter, volume confirmation, and ATR volatility filter on 4h timeframe.
# Long when price breaks above R3 with volume spike in 1w uptrend and sufficient volatility.
# Short when price breaks below S3 with volume spike in 1w downtrend and sufficient volatility.
# Exits when price returns to S3/R3 or 1w trend changes.
# Uses ATR(14) to avoid trading in low volatility periods. Volume confirmation avoids false breakouts.
# Cooldown (2 days) reduces trade frequency. Target: 15-30 trades/year. 
# 1w trend filter ensures we trade with the higher timeframe momentum, improving performance in both bull and bear markets. 
# R3/S3 levels are wider than R1/S1, reducing false breakouts and improving trade quality.