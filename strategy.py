#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Breakout at Camarilla R3/S3 levels using 1-week trend direction and volume confirmation.
# Long when 1w EMA50 rising and price breaks above R3; short when 1w EMA50 falling and price breaks below S3.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by following higher timeframe trend.
# Uses volume confirmation to avoid false breakouts and tightens trade frequency for lower fee drag.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w EMA50 for trend direction ---
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_slope = ema_50_1w - np.roll(ema_50_1w, 1)
    ema_50_1w_slope[0] = 0
    ema_50_1w_slope = pd.Series(ema_50_1w_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_slope)
    
    # --- 1d OHLC for Camarilla levels (calculate once per day) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA50 slope (50+3)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_slope_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1w EMA50 slope
        uptrend = ema_50_1w_slope_aligned[i] > 0
        downtrend = ema_50_1w_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 1w uptrend + volume surge + price breaks above R3
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 1w downtrend + volume surge + price breaks below S3
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 1w trend turns down OR price breaks below S3 (reversal)
                if downtrend or close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 1w trend turns up OR price breaks above R3 (reversal)
                if uptrend or close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals