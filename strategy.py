#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Combines Camarilla pivot breakouts with 1d trend and volume confirmation.
# Long when price breaks above R3 with 1d uptrend and volume spike; short when breaks below S3 with 1d downtrend and volume spike.
# Uses 1d trend filter to avoid whipsaw and volume spike to confirm breakout strength.
# Works in trending markets and avoids false breakouts in ranging conditions.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels (using previous day's range)
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    cam_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    cam_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (volume > 1.5 * 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure EMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get current 1d trend and Camarilla levels
        ema34_current = ema34_1d_aligned[i]
        r3_current = r3_aligned[i]
        s3_current = s3_aligned[i]
        
        trend_up = close[i] > ema34_current
        trend_down = close[i] < ema34_current
        
        if position == 0:
            # LONG: price breaks above R3 with uptrend and volume spike
            if close[i] > r3_current and trend_up and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with downtrend and volume spike
            elif close[i] < s3_current and trend_down and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price falls back below R3 or trend turns down
            if close[i] < r3_current or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above S3 or trend turns up
            if close[i] > s3_current or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals