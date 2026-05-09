#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Resistance levels: R3 = C + (H-L)*1.1/2, R4 = C + (H-L)*1.1
    # Support levels: S3 = C - (H-L)*1.1/2, S4 = C - (H-L)*1.1
    r3_1d = close_1d + range_1d * 1.1 / 2.0
    r4_1d = close_1d + range_1d * 1.1
    s3_1d = close_1d - range_1d * 1.1 / 2.0
    s4_1d = close_1d - range_1d * 1.1
    
    # Align 1d Camarilla levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get 1d data for trend filter (34-period EMA)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_1d_aligned[i]
        r3_val = r3_1d_aligned[i]
        r4_val = r4_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        s4_val = s4_1d_aligned[i]
        ema_val = ema_34_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R3 + above 1d EMA34 + volume spike
            if close[i] > r3_val and close[i] > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S3 + below 1d EMA34 + volume spike
            elif close[i] < s3_val and close[i] < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S3 or below 1d EMA34
            if close[i] < s3_val or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R3 or above 1d EMA34
            if close[i] > r3_val or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals