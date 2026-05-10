#!/usr/bin/env python3
# 6h_Camarilla_R4_S4_Breakout_12hTrend_Volume
# Hypothesis: Camarilla pivot breakouts at R4/S4 levels with 12h trend filter and volume confirmation work in both bull and bear markets.
# Breakouts at extreme pivot levels (R4/S4) indicate strong momentum, while 12h trend filter avoids counter-trend trades.
# Volume confirmation reduces false signals. Designed for low frequency (~12-37 trades/year) to minimize fee drag on 6h timeframe.

name = "6h_Camarilla_R4_S4_Breakout_12hTrend_Volume"
timeframe = "6h"
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
    
    # 12h data for Camarilla pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r4 = pivot + (range_12h * 1.1 / 2)
    r3 = pivot + (range_12h * 1.1 / 4)
    s3 = pivot - (range_12h * 1.1 / 4)
    s4 = pivot - (range_12h * 1.1 / 2)
    
    # 12h trend filter: EMA50 on 12h close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align all 12h data to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume confirmation: 24-period average on 6h (equivalent to 2x 12h periods)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above R4 with 12h uptrend and volume
            if (close[i] > r4_aligned[i] and 
                trend_12h_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S4 with 12h downtrend and volume
            elif (close[i] < s4_aligned[i] and 
                  trend_12h_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to R3 level or trend fails
            if (close[i] <= r3_aligned[i] or 
                trend_12h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to S3 level or trend fails
            if (close[i] >= s3_aligned[i] or 
                trend_12h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals