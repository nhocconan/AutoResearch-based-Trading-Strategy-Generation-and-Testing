#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla pivot levels (R3/S3) act as strong support/resistance.
# Price breaking above R3 with volume surge and aligned 1d EMA50 uptrend indicates bullish momentum.
# Price breaking below S3 with volume surge and aligned 1d EMA50 downtrend indicates bearish momentum.
# The 1d trend filter ensures we only trade in the direction of the higher timeframe trend,
# reducing false breakouts. Volume confirmation filters out low-momentum breakouts.
# Designed for low trade frequency (15-25/year) on 12h timeframe to minimize fee drag.
# Works in bull markets (breakouts continue) and bear markets (breakdowns accelerate when trend aligns).

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # --- 12h Camarilla levels (R3, S3) from previous 12h bar ---
    # Classic Camarilla formula:
    # H, L, C = high, low, close of previous period
    # R3 = C + (H - L) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    prev_12h_high = df_12h['high'].values
    prev_12h_low = df_12h['low'].values
    prev_12h_close = df_12h['close'].values
    
    camarilla_width = (prev_12h_high - prev_12h_low) * 1.1 / 2.0
    camarilla_r3 = prev_12h_close + camarilla_width
    camarilla_s3 = prev_12h_close - camarilla_width
    
    # Align 12h Camarilla levels to 12h (no shift needed as we use previous bar values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # --- 1d EMA50 for trend filter ---
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Volume confirmation (2.0x 20-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1d EMA50 and 20-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge and 1d EMA50 uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_surge and ema_50_1d_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume surge and 1d EMA50 downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_surge and ema_50_1d_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR price crosses below 1d EMA50
                if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR price crosses above 1d EMA50
                if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals