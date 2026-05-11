#!/usr/bin/env python3
# 12h_1d_Camarilla_R3_S3_Breakout_TrendFilter_v1
# Hypothesis: Uses Camarilla R3/S3 levels from 1d timeframe for breakout entries on 12h chart.
# Long when price breaks above R3 with volume confirmation and above 1d EMA34 trend.
# Short when price breaks below S3 with volume confirmation and below 1d EMA34 trend.
# Designed for low trade frequency by requiring confluence of price level breakout,
# volume spike, and trend alignment. Works in both bull and bear markets by following
# the higher timeframe trend (1d EMA34).

name = "12h_1d_Camarilla_R3_S3_Breakout_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla R3 and S3 levels from 1d data ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- Trend Filter (EMA34 on 1d close) ---
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume, above 1d EMA34
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with volume, below 1d EMA34
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of trend
            if position == 1:
                # Exit long: price breaks below Camarilla S3
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above Camarilla R3
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals