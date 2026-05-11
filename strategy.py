#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels, trend, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla levels (R3, S3)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + range_ * 1.1 / 4
    camarilla_s3 = prev_close - range_ * 1.1 / 4
    
    # Calculate EMA34 on daily close for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for EMA34 and volume MA to be valid
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + above daily EMA34 + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + below daily EMA34 + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below S3 (reversion to mean)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above R3 (reversion to mean)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals