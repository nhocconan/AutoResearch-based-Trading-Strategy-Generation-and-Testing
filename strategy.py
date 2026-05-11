#!/usr/bin/env python3
# 1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Weekly trend (1w EMA50) + Daily Camarilla R3/S3 breakout with volume confirmation
# In bull markets: 1w uptrend + daily breakout above R3 captures momentum.
# In bear markets: 1w downtrend + daily breakdown below S3 captures accelerated moves.
# Volume filter ensures breakouts have conviction. Target: 15-25 trades/year.

name = "1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA50 for trend filter ---
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Daily Camarilla levels (R3, S3) from previous daily bar ---
    prev_daily_high = df_1w['high'].values  # Wait, using wrong df - fix below
    
    # Fix: Need daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_daily_high = df_1d['high'].values
    prev_daily_low = df_1d['low'].values
    prev_daily_close = df_1d['close'].values
    
    camarilla_width = (prev_daily_high - prev_daily_low) * 1.1 / 2.0
    camarilla_r3 = prev_daily_close + camarilla_width
    camarilla_s3 = prev_daily_close - camarilla_width
    
    # Align daily Camarilla levels to daily timeframe (identity since same TF)
    camarilla_r3_aligned = camarilla_r3  # Already at daily frequency
    camarilla_s3_aligned = camarilla_s3
    
    # --- Volume confirmation (2.0x 20-period average on daily) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for weekly EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge and weekly uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                ema_50_1w_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume surge and weekly downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  ema_50_1w_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR weekly EMA50 turns down
                if (close[i] < camarilla_s3_aligned[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR weekly EMA50 turns up
                if (close[i] > camarilla_r3_aligned[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals