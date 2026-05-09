#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily close for Camarilla calculation (use previous day's close)
    close_1d = df_1d['close'].values
    # Calculate Camarilla levels: based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using previous day's data)
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # where C, H, L are from previous day
    cam_r4 = close_1d_prev + (high_1d - low_1d) * 1.1 / 2
    cam_r3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    cam_s3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    cam_s4 = close_1d_prev - (high_1d - low_1d) * 1.1 / 2
    
    # Daily volume average for volume spike filter
    vol_series_1d = pd.Series(df_1d['volume'])
    vol_avg_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_1d_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    cam_r4_6h = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_r3_6h = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_6h = align_htf_to_ltf(prices, df_1d, cam_s3)
    cam_s4_6h = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # wait for EMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_12h_6h[i]) or np.isnan(vol_avg_1d_6h[i]) or
            np.isnan(cam_r4_6h[i]) or np.isnan(cam_r3_6h[i]) or
            np.isnan(cam_s3_6h[i]) or np.isnan(cam_s4_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_12h_6h[i]
        vol_avg = vol_avg_1d_6h[i]
        vol_spike = volume[i] > vol_avg * 2.0  # Volume spike filter
        
        if position == 0:
            # Long: break above R3 with volume spike and above 12h trend
            if close[i] > cam_r3_6h[i] and vol_spike and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume spike and below 12h trend
            elif close[i] < cam_s3_6h[i] and vol_spike and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below R3 or trend reversal
            if close[i] < cam_r3_6h[i] or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above S3 or trend reversal
            if close[i] > cam_s3_6h[i] or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals