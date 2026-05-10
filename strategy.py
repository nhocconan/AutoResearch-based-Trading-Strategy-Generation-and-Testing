#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Breakout from Camarilla R3/S3 levels with 12h trend and volume confirmation.
# Works in bull/bear by following 12h trend direction, entering only on strong volume breakouts.
# Camarilla levels provide institutional support/resistance; volume confirms institutional participation.
# Target: 20-35 trades/year per symbol.

name = "4H_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Calculate Camarilla levels for each bar using previous day's OHLC
    # We need daily OHLC to compute Camarilla levels
    # Get daily data first
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC arrays
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    daily_range = daily_high - daily_low
    camarilla_r3 = daily_close + daily_range * 1.1 / 2
    camarilla_s3 = daily_close - daily_range * 1.1 / 2
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_12h_up = close_12h > ema20_12h
    trend_12h_down = close_12h < ema20_12h
    
    # Align 12h trend to 4h
    trend_12h_up_4h = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_4h = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or
            np.isnan(trend_12h_up_4h[i]) or np.isnan(trend_12h_down_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above R3 with 12h uptrend and volume
            if trend_12h_up_4h[i] > 0.5 and volume_confirm:
                if close[i] > camarilla_r3_4h[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: price breaks below S3 with 12h downtrend and volume
            elif trend_12h_down_4h[i] > 0.5 and volume_confirm:
                if close[i] < camarilla_s3_4h[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price moves back below R3 or trend changes
            if close[i] < camarilla_r3_4h[i] or trend_12h_up_4h[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves back above S3 or trend changes
            if close[i] > camarilla_s3_4h[i] or trend_12h_down_4h[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals