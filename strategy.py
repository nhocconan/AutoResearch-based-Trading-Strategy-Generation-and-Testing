#!/usr/bin/env python3
# 12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: 12h timeframe with daily trend filter and daily Camarilla R3/S3 breakouts.
# Uses volume confirmation to ensure breakout strength. Designed for fewer trades
# (target 12-37/year) to reduce fee drag while capturing significant moves in
# both bull and bear markets via trend alignment.

name = "12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily EMA50 for trend filter ---
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Daily Camarilla levels (R3, S3) from previous day ---
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla width: (H-L) * 1.1 / 2
    camarilla_width = (prev_high - prev_low) * 1.1 / 2.0
    camarilla_r3 = prev_close + camarilla_width
    camarilla_s3 = prev_close - camarilla_width
    
    # Align daily levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # --- Volume confirmation: 2x 24-period average on 12h ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: 50 for daily EMA50 + 24 for volume MA
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
        
        # Volume surge condition
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge and daily uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                ema_50_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume surge and daily downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  ema_50_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR daily EMA50 turns down
                if (close[i] < camarilla_s3_aligned[i] or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR daily EMA50 turns up
                if (close[i] > camarilla_r3_aligned[i] or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals