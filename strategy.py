#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Price breaking Camarilla R3/S3 levels with 12h trend and volume confirmation provides high-probability breakouts. Camarilla levels act as natural support/resistance, while 12h trend ensures alignment with higher timeframe momentum. Volume filters false breakouts. Designed for low frequency (20-40 trades/year) to minimize fee drag in both bull and bear markets.

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R3 and S3 levels
    R3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    S3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
    # 12h trend: EMA50 on close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume confirmation: 20-period average on 4h
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above R3 with 12h uptrend and volume
            if (close[i] > R3_aligned[i] and 
                trend_12h_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with 12h downtrend and volume
            elif (close[i] < S3_aligned[i] and 
                  trend_12h_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to S3 level or trend fails
            if (close[i] < S3_aligned[i] or 
                trend_12h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to R3 level or trend fails
            if (close[i] > R3_aligned[i] or 
                trend_12h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals