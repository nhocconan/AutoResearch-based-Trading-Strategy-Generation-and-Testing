#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter and context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for Camarilla calculation
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    prev_close_12h = df_12h['close'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3
    camarilla_r3 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 2
    camarilla_s3 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # 12h EMA trend filter (50-period)
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume spike: current volume > 2.5 * 20-period average (on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Wait for EMA and volume MA warmup
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 12h uptrend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_12h_aligned[i] and 
                volume[i] > 2.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 12h downtrend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_12h_aligned[i] and 
                  volume[i] > 2.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below R3 or drops below 12h EMA
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above S3 or rises above 12h EMA
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals