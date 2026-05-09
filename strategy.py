# 4H_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeS
name = "4H_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_r4 = np.zeros(len(df_1d))
    camarilla_s4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i >= 1:  # Need previous day's data
            high_prev = high_1d[i-1]
            low_prev = low_1d[i-1]
            close_prev = close_1d[i-1]
            rang = high_prev - low_prev
            camarilla_r3[i] = close_prev + rang * 1.1 / 4
            camarilla_s3[i] = close_prev - rang * 1.1 / 4
            camarilla_r4[i] = close_prev + rang * 1.1 / 2
            camarilla_s4[i] = close_prev - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    volume_ma = np.zeros(n)
    for i in range(20, n):
        volume_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5 if i >= 20 else False
        
        if position == 0:
            # Enter long: price above R3 + trend up + volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and 
                ema50_12h_aligned[i] > camarilla_s3_aligned[i] and  # Trend filter: EMA50 > S3
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price below S3 + trend down + volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema50_12h_aligned[i] < camarilla_r3_aligned[i] and  # Trend filter: EMA50 < R3
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below S3 or trend reversal
            if (close[i] < camarilla_s3_aligned[i] or 
                ema50_12h_aligned[i] < camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above R3 or trend reversal
            if (close[i] > camarilla_r3_aligned[i] or 
                ema50_12h_aligned[i] > camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals