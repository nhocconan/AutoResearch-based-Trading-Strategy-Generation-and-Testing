# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 12-hour timeframe, price breaking above Camarilla R3 or below S3 with
# daily trend alignment and volume spike captures strong momentum moves. This strategy
# works in both bull and bear markets by following the daily trend direction while
# using volume to filter false breakouts. Camarilla levels provide natural support/resistance
# based on prior day's range, and breaking R3/S3 indicates institutional interest.
# Timeframe: 12h (slower = fewer trades, less fee drag). Target: 20-50 trades/year.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H = high, L = low, C = close of previous day
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500)
    # S3 = C - ((H-L) * 1.2500), S4 = C - ((H-L) * 1.5000)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero and NaN for first day
    valid_prev = (~np.isnan(prev_high)) & (~np.isnan(prev_low)) & (~np.isnan(prev_close))
    high_low_diff = np.where(valid_prev, prev_high - prev_low, 0.0)
    
    camarilla_r3 = np.where(valid_prev, prev_close + (high_low_diff * 1.25), np.nan)
    camarilla_s3 = np.where(valid_prev, prev_close - (high_low_diff * 1.25), np.nan)
    
    # Align Camarilla levels to 12h timeframe (they change only at daily boundaries)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 2-period volume MA on 12h chart (approx 1 day)
    volume_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50_1d (50) and volume MA (2)
    start_idx = max(50, 2)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R3, daily uptrend, volume confirmation
            if (close[i] > camarilla_r3_aligned[i]) and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3, daily downtrend, volume confirmation
            elif (close[i] < camarilla_s3_aligned[i]) and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal) or trend breaks
            if (close[i] < camarilla_s3_aligned[i]) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 (reversal) or trend breaks
            if (close[i] > camarilla_r3_aligned[i]) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3