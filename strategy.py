#!/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA for weekly trend
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Camarilla pivot levels from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each weekly bar
    camarilla_r3 = np.zeros(len(df_1w))
    camarilla_s3 = np.zeros(len(df_1w))
    
    for i in range(len(df_1w)):
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            continue
        # Use previous week's data for current week's levels
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        camarilla_r3[i] = pc + (ph - pl) * 1.1 / 4
        camarilla_s3[i] = pc - (ph - pl) * 1.1 / 4
    
    # Align weekly data to daily
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume filter: 20-period average on daily
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above R3 with volume and weekly uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                ema_50_aligned[i] > ema_50_aligned[i-1]):  # Rising EMA = uptrend
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume and weekly downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  ema_50_aligned[i] < ema_50_aligned[i-1]):  # Falling EMA = downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to the opposite Camarilla level or weekly EMA
            if position == 1:
                # Exit long: price returns below S3 or weekly EMA turns down
                if (close[i] < camarilla_s3_aligned[i]) or (ema_50_aligned[i] < ema_50_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns above R3 or weekly EMA turns up
                if (close[i] > camarilla_r3_aligned[i]) or (ema_50_aligned[i] > ema_50_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals