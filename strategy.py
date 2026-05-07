#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels for current day (based on previous day)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = prev_close + prev_range * 1.1 / 4
    camarilla_s3 = prev_close - prev_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily trend filter: 34-period EMA on close
    daily_ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = np.where(vol_ma > 0, volume / vol_ma, 1.0) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(daily_ema34_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below daily EMA34
        uptrend = close[i] > daily_ema34_aligned[i]
        downtrend = close[i] < daily_ema34_aligned[i]
        
        if position == 0:
            # Long breakout: price > R3 with volume spike in uptrend
            long_condition = (close[i] > r3_aligned[i]) and vol_spike[i] and uptrend
            # Short breakdown: price < S3 with volume spike in downtrend
            short_condition = (close[i] < s3_aligned[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below R3 or trend turns down
            if (close[i] < r3_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above S3 or trend turns up
            if (close[i] > s3_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals