#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily Camarilla pivot levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels
    close_prev = daily_close
    range_ = daily_high - daily_low
    
    # Resistance levels
    r1 = close_prev + (range_ * 1.0 / 12)
    r2 = close_prev + (range_ * 2.0 / 12)
    r3 = close_prev + (range_ * 3.0 / 12)
    r4 = close_prev + (range_ * 4.0 / 12)
    
    # Support levels
    s1 = close_prev - (range_ * 1.0 / 12)
    s2 = close_prev - (range_ * 2.0 / 12)
    s3 = close_prev - (range_ * 3.0 / 12)
    s4 = close_prev - (range_ * 4.0 / 12)
    
    # Align Camarilla levels to 4h timeframe (with 1-bar delay for completed daily bar)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: spike above 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(r4_4h[i]) or np.isnan(s4_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume spike confirmation
        
        if position == 0:
            # Long: price breaks above S1, 12h uptrend (price > EMA50), volume spike
            if (close[i] > s1_4h[i] and 
                close[i] > ema_50_4h[i] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R1, 12h downtrend (price < EMA50), volume spike
            elif (close[i] < r1_4h[i] and 
                  close[i] < ema_50_4h[i] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S2 or trend reversal
            if close[i] < s2_4h[i] or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R2 or trend reversal
            if close[i] > r2_4h[i] or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals