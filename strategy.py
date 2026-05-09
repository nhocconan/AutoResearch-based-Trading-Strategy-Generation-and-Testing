#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 10-period EMA on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h CAMARILLA R3 and S3 from previous 4h bar's OHLC
    prev_4h_high = df_4h['high'].shift(1).values
    prev_4h_low = df_4h['low'].shift(1).values
    prev_4h_close = df_4h['close'].shift(1).values
    
    # Camarilla formula for R3 and S3
    range_4h = prev_4h_high - prev_4h_low
    camarilla_mult = 1.1 / 12  # ~0.0916667
    r3_4h = prev_4h_close + range_4h * camarilla_mult * 3
    s3_4h = prev_4h_close - range_4h * camarilla_mult * 3
    
    # Align Camarilla levels to 4h timeframe (already aligned to 4h bars)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 4)  # Need 30 for 1w EMA and 4 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(r3_4h_aligned[i]) or 
            np.isnan(s3_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_10_1w_aligned[i]
        r3_level = r3_4h_aligned[i]
        s3_level = s3_4h_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price breaks above R3 with volume AND price > 1w EMA10 (uptrend)
            if close[i] > r3_level and vol > 2.5 * vol_ma_val and close[i] > ema_1w:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S3 with volume AND price < 1w EMA10 (downtrend)
            elif close[i] < s3_level and vol > 2.5 * vol_ma_val and close[i] < ema_1w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below R3 OR trend reverses (price < 1w EMA10)
            if close[i] < r3_level or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above S3 OR trend reverses (price > 1w EMA10)
            if close[i] > s3_level or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals