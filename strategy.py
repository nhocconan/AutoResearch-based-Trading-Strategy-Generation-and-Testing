#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above R3 AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below S3 AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit when price crosses Camarilla H4/L4 levels (mean reversion to median)
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 30-60 trades/year on 4h (120-240 total over 4 years).
# 12h EMA50 provides smoother trend filter than 1d EMA34, reducing whipsaw in sideways markets.
# Volume spike at 1.8x (slightly lower than 2.0) increases trade frequency while maintaining quality.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels (prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract daily OHLC values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily OHLC to 4h timeframe (each value represents the prior day's close)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate Camarilla levels for each 4h bar based on prior day's OHLC
    # Camarilla R3/S3 and H4/L4 levels
    daily_range = daily_high_aligned - daily_low_aligned
    camarilla_h4 = daily_close_aligned + daily_range * 1.1 / 2
    camarilla_l4 = daily_close_aligned - daily_range * 1.1 / 2
    camarilla_r3 = daily_close_aligned + daily_range * 1.1 / 4
    camarilla_s3 = daily_close_aligned - daily_range * 1.1 / 4
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_l4[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_12h_aligned[i]
        
        # Camarilla levels
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below H4 (mean reversion to median)
            if curr_close < h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above L4 (mean reversion to median)
            if curr_close > l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 12h EMA50 AND volume confirmation
            if curr_close > r3 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 12h EMA50 AND volume confirmation
            elif curr_close < s3 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals