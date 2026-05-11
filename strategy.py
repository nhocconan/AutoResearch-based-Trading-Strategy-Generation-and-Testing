#!/usr/bin/env python3
# 4h_4C_Breakout_1dTrend_1dVol
# Hypothesis: Breakout above/below prior day's high/low with 1d trend and volume confirmation.
# Long when price > prior day high, 1d EMA20 rising, and volume > 1d average.
# Short when price < prior day low, 1d EMA20 falling, and volume > 1d average.
# Works in bull markets (riding breakouts up) and bear markets (riding breakouts down).
# Uses 1d timeframe for trend/volume, 4h for execution to reduce noise and overtrading.

name = "4h_4C_Breakout_1dTrend_1dVol"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA20 for trend direction ---
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_slope = ema_20_1d - np.roll(ema_20_1d, 1)
    ema_20_1d_slope[0] = 0
    ema_20_1d_slope = pd.Series(ema_20_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values  # smooth slope
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_20_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d_slope)
    
    # --- 1d average volume ---
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # --- Prior day high/low ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    prior_high_1d = np.roll(high_1d, 1)
    prior_low_1d = np.roll(low_1d, 1)
    prior_high_1d[0] = 0  # will be ignored due to warmup
    prior_low_1d[0] = 0
    prior_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_high_1d)
    prior_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_low_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA20 (20) and slope (20+3)
    start_idx = 23
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_1d_slope_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(prior_high_1d_aligned[i]) or
            np.isnan(prior_low_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1d EMA20 slope
        uptrend = ema_20_1d_slope_aligned[i] > 0
        downtrend = ema_20_1d_slope_aligned[i] < 0
        
        # Volume confirmation
        vol_surge = volume[i] > vol_avg_1d_aligned[i]
        
        if position == 0:
            if uptrend and vol_surge:
                # Long: 1d uptrend + volume surge + price above prior day high
                if close[i] > prior_high_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge:
                # Short: 1d downtrend + volume surge + price below prior day low
                if close[i] < prior_low_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 1d trend turns down OR price crosses below prior day low
                if downtrend or close[i] < prior_low_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 1d trend turns up OR price crosses above prior day high
                if uptrend or close[i] > prior_high_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals