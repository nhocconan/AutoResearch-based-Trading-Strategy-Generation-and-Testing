#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate EMA21 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 1d data for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + camarilla_range * 1.50
    s3 = close_1d - camarilla_range * 1.50
    
    # Align Camarilla R3 and S3 to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike filter: current 1h volume > 1.5 * 24-period average (24h)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_4h = ema_21_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close > R3 and price above 4h EMA21 with volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: Close < S3 and price below 4h EMA21 with volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Close < S3 or trend breaks (price < 4h EMA21)
            if close[i] < s3_aligned[i] or close[i] < ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Close > R3 or trend breaks (price > 4h EMA21)
            if close[i] > r3_aligned[i] or close[i] > ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals