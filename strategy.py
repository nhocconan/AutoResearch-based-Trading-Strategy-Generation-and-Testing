#!/usr/bin/env python3
# 4h_12h_Camarilla_R3_S3_Breakout_Trend_Volume_v2
# Hypothesis: 12h Camarilla pivot levels (R3/S3) act as strong support/resistance levels.
# Price breaking above R3 with volume surge and aligned 12h EMA50 uptrend indicates bullish momentum.
# Price breaking below S3 with volume surge and aligned 12h EMA50 downtrend indicates bearish momentum.
# The 12h trend filter ensures we only trade in the direction of the higher timeframe trend,
# reducing false breakouts. Volume confirmation filters out low-momentum breakouts.
# This version reduces trade frequency by requiring a stronger volume surge (3.0x) and 
# adding a minimum holding period of 12 bars (3 days) to reduce whipsaw and fee drag.
# Designed for low trade frequency (15-30/year) to improve test generalization.
# Works in bull markets (breakouts continue) and bear markets (breakdowns accelerate when trend aligns).

name = "4h_12h_Camarilla_R3_S3_Breakout_Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    
    # Align 12h Camarilla levels to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # --- 12h EMA50 for trend filter ---
    ema_50_12h = pd.Series(prev_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Volume confirmation (3.0x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: enough for 12h EMA50 and 20-period volume MA
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 3.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge and 12h EMA50 uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_surge and ema_50_12h_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S3 with volume surge and 12h EMA50 downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_surge and ema_50_12h_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        else:
            bars_since_entry += 1
            # Minimum holding period: 12 bars (3 days)
            if bars_since_entry < 12:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
                continue
            
            if position == 1:
                # Exit long: price drops below S3 OR price crosses below 12h EMA50
                if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR price crosses above 12h EMA50
                if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals