#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
# Uses 1d Camarilla pivot levels (R3/S3) for institutional breakout detection.
# Filtered by 1d EMA50 trend and volume > 2.5x 20-period median to reduce false signals.
# R3/S3 breaks work in both bull (buy R3 with uptrend) and bear (sell S3 with downtrend).
# Discrete position sizing (0.30) balances return and drawdown. Target: 75-200 trades over 4 years.

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    rang = high_1d - low_1d
    r3 = pivot + rang * 1.1 / 2
    s3 = pivot - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50 and volume median
    start_idx = max(50, 20) + 1  # 51
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.5)
        
        # Camarilla breakout conditions (R3/S3 = institutional breakout levels)
        breakout_up = curr_close > r3_aligned[i]   # break above R3
        breakout_down = curr_close < s3_aligned[i]  # break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND uptrend AND volume confirmation
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Breakdown below S3 AND downtrend AND volume confirmation
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S3 (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on breakout above R3 (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals