#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA200 trend filter and volume confirmation.
# Long when price breaks above R3 level, close > 12h EMA200, and volume > 2.0x 20-bar avg.
# Short when price breaks below S3 level, close < 12h EMA200, and volume > 2.0x 20-bar avg.
# Exit when price moves back inside the R3-S3 range.
# Uses Camarilla pivot levels from 1d timeframe for precise intraday structure.
# 12h EMA200 provides strong higher timeframe trend filter to avoid counter-trend trades in bear markets.
# Volume confirmation with 2.0x threshold reduces false breakouts.
# Discrete position sizing at ±0.25 to balance performance and fee drag.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion exits.
# Strong trend filter (EMA200) reduces overtrading and improves signal quality.

name = "4h_Camarilla_R3S3_Breakout_12hEMA200_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    pivot_point = (high_1d + low_1d + close_1d_vals) / 3
    camarilla_r3 = pivot_point + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = pivot_point - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 210:  # Need enough for EMA200
        return np.zeros(n)
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 210  # warmup for EMA200 and pivot points
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_200_12h = ema_200_12h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, close > 12h EMA200, volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_200_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, close < 12h EMA200, volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_200_12h and 
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