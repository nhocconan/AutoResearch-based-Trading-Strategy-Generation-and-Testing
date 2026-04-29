#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3S3 Breakout with 12h EMA50 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R3 level AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 level AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit when price reverses back to Camarilla pivot (PP) level OR momentum fails (price crosses EMA34 opposite direction)
# Uses discrete position sizing (0.25) to minimize fee drag. Camarilla levels provide precise intraday support/resistance,
# 12h EMA50 filters counter-trend moves, volume confirmation ensures breakout strength.
# Works in bull markets (buying breakouts) and bear markets (selling breakdowns).

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (using daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla R3 = close + 1.1 * (high - low) / 2
    # Camarilla S3 = close - 1.1 * (high - low) / 2
    # Camarilla PP = (high + low + close) / 3
    diff_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + 1.1 * diff_1d / 2.0
    camarilla_s3_1d = close_1d - 1.1 * diff_1d / 2.0
    camarilla_pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp_1d)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 alignment warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(close[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_pp = camarilla_pp_aligned[i]
        curr_ema50 = ema_50_12h_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Camarilla PP OR price crosses below EMA50 (trend fails)
            if curr_close < curr_pp or curr_close < curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla PP OR price crosses above EMA50 (trend fails)
            if curr_close > curr_pp or curr_close > curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 12h EMA50 AND volume confirmation
            if curr_close > curr_r3 and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 12h EMA50 AND volume confirmation
            elif curr_close < curr_s3 and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals