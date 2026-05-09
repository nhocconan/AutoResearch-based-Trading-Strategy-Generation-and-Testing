#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Get 12h data for trend and daily for Camarilla pivot
    df_12h = get_htf_data(prices, '12h')
    df_d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_d) < 50:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla pivot (R3, S3)
    daily_high = df_d['high'].values
    daily_low = df_d['low'].values
    daily_close = df_d['close'].values
    
    # Calculate Camarilla pivot levels (R3, S3)
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 4
    # S3 = Pivot - (H - L) * 1.1 / 4
    pivot = (daily_high + daily_low + daily_close) / 3
    r3 = pivot + (daily_high - daily_low) * 1.1 / 4
    s3 = pivot - (daily_high - daily_low) * 1.1 / 4
    
    # 12h EMA(50) for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_series = pd.Series(volume)
    vol_ma50 = vol_series.rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3) or np.isnan(s3) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma50[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume and above 12h EMA trend
            if close[i] > r3 and vol_ok and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume and below 12h EMA trend
            elif close[i] < s3 and vol_ok and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price closes below S3 (reversion to mean)
            if close[i] < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above R3 (reversion to mean)
            if close[i] > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals