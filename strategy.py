#!/usr/bin/env python3
# 6h_12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: On 6h timeframe, use 12h Camarilla R3/S3 levels as breakout triggers and 1d close for trend filter.
# Price breaking above 12h R3 with volume surge and 1d close above 12h EMA50 indicates bullish momentum.
# Price breaking below 12h S3 with volume surge and 1d close below 12h EMA50 indicates bearish momentum.
# The 1d trend filter ensures we only trade in the direction of the higher timeframe trend (1d),
# reducing false breakouts. Volume confirmation filters out low-momentum breakouts.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 6s timeframe.

name = "6h_12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for Camarilla levels and EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter (close price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h Camarilla levels (R3, S3) from previous 12h bar ---
    prev_12h_high = df_12h['high'].values
    prev_12h_low = df_12h['low'].values
    prev_12h_close = df_12h['close'].values
    
    camarilla_width = (prev_12h_high - prev_12h_low) * 1.1 / 2.0
    camarilla_r3 = prev_12h_close + camarilla_width
    camarilla_s3 = prev_12h_close - camarilla_width
    
    # Align 12h Camarilla levels to 6h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # --- 12h EMA50 for trend filter (using 1d close as proxy for trend) ---
    ema_50_12h = pd.Series(prev_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- 1d close for trend filter ---
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # --- Volume confirmation (2.0x 20-period average on 6h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 12h EMA50 and 20-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 12h R3 with volume surge and 1d close above 12h EMA50
            if close[i] > camarilla_r3_aligned[i] and volume_surge and close_1d_aligned[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h S3 with volume surge and 1d close below 12h EMA50
            elif close[i] < camarilla_s3_aligned[i] and volume_surge and close_1d_aligned[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below 12h S3 OR 1d close drops below 12h EMA50
                if close[i] < camarilla_s3_aligned[i] or close_1d_aligned[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above 12h R3 OR 1d close rises above 12h EMA50
                if close[i] > camarilla_r3_aligned[i] or close_1d_aligned[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals