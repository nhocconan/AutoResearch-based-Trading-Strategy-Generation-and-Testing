#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 level, close > 1d EMA34, and volume > 1.5x 20-bar avg.
# Short when price breaks below S3 level, close < 1d EMA34, and volume > 1.5x 20-bar avg.
# Exit when price moves back inside the R3-S3 range.
# Uses 12h primary timeframe to reduce trade frequency (target: 50-150 trades over 4 years).
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation with 1.5x threshold reduces false breakouts while maintaining sufficient trades.
# Discrete position sizing at ±0.25 to balance performance and fee drag.
# Designed to work in both bull and bear markets by combining breakout logic with trend and volume filters.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    pivot_point = (high_1d + low_1d + close_1d_vals) / 3
    camarilla_r3 = pivot_point + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = pivot_point - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for EMA34 and pivot points
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, close > 1d EMA34, volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, close < 1d EMA34, volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves back inside R3-S3 range (below R3)
            if curr_close < curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price moves back inside R3-S3 range (above S3)
            if curr_close > curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals