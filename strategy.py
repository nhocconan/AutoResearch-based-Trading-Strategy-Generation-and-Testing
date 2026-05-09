#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Camarilla_R3S3_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA10 for trend filter
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate daily Camarilla levels (R3, S3)
    # Camarilla formula: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+CLOSE)/3 of previous day
    
    # We need previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate pivot point (CP)
    cp = (prev_high + prev_low + prev_close) / 3.0
    # Calculate range
    rang = prev_high - prev_low
    
    # Calculate R3 and S3
    r3 = cp + (rang * 1.1 / 4.0)
    s3 = cp - (rang * 1.1 / 4.0)
    
    # Align weekly trend to daily
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(r3[i]) or np.isnan(s3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_avg_today = vol_avg_20[i]
        vol_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_current > 1.5 * vol_avg_today
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and weekly uptrend
            if price > r3[i] and vol_confirmed and price > ema10_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S3 with volume and weekly downtrend
            if price < s3[i] and vol_confirmed and price < ema10_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price breaks below S3 or trend changes
            if price < s3[i] or price < ema10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or trend changes
            if price > r3[i] or price > ema10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals