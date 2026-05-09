#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Camarilla formula
    cam = (prev_day_high - prev_day_low) * 1.1 / 12
    r3 = prev_day_close + cam * 1.1
    r2 = prev_day_close + cam * 0.55
    r1 = prev_day_close + cam * 0.275
    s1 = prev_day_close - cam * 0.275
    s2 = prev_day_close - cam * 0.55
    s3 = prev_day_close - cam * 1.1
    
    # Align daily levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 50-period EMA on daily close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 50 for daily EMA and 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_level = r3_aligned[i]
        r2_level = r2_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        s2_level = s2_aligned[i]
        s3_level = s3_aligned[i]
        atr_1d = atr_1d_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Only trade when volatility is sufficient (avoid choppy markets)
        vol_filter = atr_1d > 0  # Always true, but keeps structure for potential adjustment
        
        if position == 0:
            # Enter long: Price breaks above R3 with volume AND price > daily EMA50 (strong uptrend)
            if close[i] > r3_level and vol > 2.0 * vol_ma_val and close[i] > ema_1d:
                signals[i] = 0.30
                position = 1
            # Enter short: Price breaks below S3 with volume AND price < daily EMA50 (strong downtrend)
            elif close[i] < s3_level and vol > 2.0 * vol_ma_val and close[i] < ema_1d:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below R2 OR trend reverses (price < daily EMA50)
            if close[i] < r2_level or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: Price breaks above S2 OR trend reverses (price > daily EMA50)
            if close[i] > s2_level or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals