#!/usr/bin/env python3
# 6H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT
# Hypothesis: Trade breakouts from Camarilla R3/S3 levels in direction of 1d trend with volume confirmation.
# Long when: price breaks above R3 and closes above it, 1d uptrend, volume > 2x average.
# Short when: price breaks below S3 and closes below it, 1d downtrend, volume > 2x average.
# Uses 12h EMA50 as secondary trend filter to avoid chop.
# Works in bull/bear by following 1d trend and using volume to confirm institutional interest.
# Target: 15-30 trades/year per symbol.

name = "6H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
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
    
    # 6h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 6h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous 12h bar
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            if range_val > 0:
                R3 = prev_close + (range_val * 1.1000 / 4)
                S3 = prev_close - (range_val * 1.1000 / 4)
                
                vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
                volume_spike = vol_ratio > 2.0
                
                daily_up = daily_uptrend_aligned[i] > 0.5
                daily_down = daily_downtrend_aligned[i] > 0.5
                trend_up = trend_12h_up_aligned[i] > 0.5
                trend_down = trend_12h_down_aligned[i] > 0.5
                
                if position == 0:
                    # Enter long: price breaks above R3, 1d uptrend, 12h uptrend, volume spike
                    if close[i] > R3 and daily_up and trend_up and volume_spike:
                        signals[i] = 0.25
                        position = 1
                    # Enter short: price breaks below S3, 1d downtrend, 12h downtrend, volume spike
                    elif close[i] < S3 and daily_down and trend_down and volume_spike:
                        signals[i] = -0.25
                        position = -1
                
                elif position == 1:
                    # Exit: price closes below R3 or trend changes
                    if close[i] < R3 or not daily_up or not trend_up:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                
                elif position == -1:
                    # Exit: price closes above S3 or trend changes
                    if close[i] > S3 or not daily_down or not trend_down:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals