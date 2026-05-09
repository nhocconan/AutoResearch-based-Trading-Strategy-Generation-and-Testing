#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_1dTrend_Volume_143148"
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
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (Camarilla)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Camarilla formula: H-L range
    range_w = high_w - low_w
    # Resistance levels (Camarilla)
    r3_w = close_w + range_w * 1.1 / 2
    r4_w = close_w + range_w * 1.1
    s3_w = close_w - range_w * 1.1 / 2
    s4_w = close_w - range_w * 1.1
    
    # Align weekly Camarilla levels to 12h timeframe
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily close for trend filter
    close_d = df_d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Volume spike filter: current volume > 2.0 * 30-period average (tighter threshold)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Need enough data for EMA50 (daily) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema50_d_aligned[i]) or 
            np.isnan(r3_w_aligned[i]) or
            np.isnan(r4_w_aligned[i]) or
            np.isnan(s3_w_aligned[i]) or
            np.isnan(s4_w_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_d_val = ema50_d_aligned[i]
        r3_w_val = r3_w_aligned[i]
        r4_w_val = r4_w_aligned[i]
        s3_w_val = s3_w_aligned[i]
        s4_w_val = s4_w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price breaks above R3 + daily uptrend + volume spike
            if close[i] > r3_w_val and close[i] > ema50_d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S3 + daily downtrend + volume spike
            elif close[i] < s3_w_val and close[i] < ema50_d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below R3 or daily trend turns down
            if close[i] < r3_w_val or close[i] < ema50_d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above S3 or daily trend turns up
            if close[i] > s3_w_val or close[i] > ema50_d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals