#!/usr/bin/env python3
# 4H_1D_1W_Camarilla_R3_S3_Breakout_1DTrend_Volume
# Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R3 with 1-day uptrend and volume spike.
# Enter short when price breaks below Camarilla S3 with 1-day downtrend and volume spike.
# Trend filter from 1-day EMA34 slope to avoid counter-trend trades. Volume confirmation reduces false breakouts.
# Uses tight entry conditions to limit trades (target: 20-40/year) and avoid fee drag.
# Works in bull markets via breakouts and in bear via short breakdowns with trend alignment.

name = "4H_1D_1W_Camarilla_R3_S3_Breakout_1DTrend_Volume"
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
    
    # Get 1-day data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R3, S3
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # 1-day trend: EMA34 slope > 0 for uptrend
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_slope = np.diff(ema_34, prepend=np.nan)  # today - yesterday
    trend_up = ema34_slope > 0
    
    # Volume confirmation: current volume > 2x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 2.0)
    
    # Align Camarilla levels and trend to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + 1-day uptrend + volume confirmation
            if close[i] > R3_aligned[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + 1-day downtrend + volume confirmation
            elif close[i] < S3_aligned[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (mean reversion) or trend turns down
            if close[i] < S3_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 (mean reversion) or trend turns up
            if close[i] > R3_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals