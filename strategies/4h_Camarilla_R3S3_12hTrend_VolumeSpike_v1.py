#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_12hTrend_VolumeSpike_v1"
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
    
    # Get 12h data for Camarilla pivot levels (primary levels)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (standard formula)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Standard Camarilla: P = (H + L + C)/3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Resistance/Support levels (Camarilla specific)
    r3_12h = close_12h + (high_12h - low_12h) * 1.1 / 2
    s3_12h = close_12h - (high_12h - low_12h) * 1.1 / 2
    r4_12h = close_12h + (high_12h - low_12h) * 1.1
    s4_12h = close_12h - (high_12h - low_12h) * 1.1
    
    # Align 12h Camarilla levels to 4h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Get 12h trend filter: EMA50 on 12h close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current 4h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or
            np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_12h_aligned[i]
        r3_val = r3_12h_aligned[i]
        s3_val = s3_12h_aligned[i]
        r4_val = r4_12h_aligned[i]
        s4_val = s4_12h_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R3 + above 12h EMA50 + volume spike
            if close[i] > r3_val and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S3 + below 12h EMA50 + volume spike
            elif close[i] < s3_val and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S3 or below 12h EMA50
            if close[i] < s3_val or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R3 or above 12h EMA50
            if close[i] > r3_val or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals