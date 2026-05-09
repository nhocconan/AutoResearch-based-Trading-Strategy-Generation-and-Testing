#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeS"
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
    
    # Get 1d data for primary trend and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard Camarilla: P = (H + L + C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Resistance/Support levels (Camarilla specific)
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    r2_1d = close_1d + (high_1d - low_1d) * 1.1 / 6
    s2_1d = close_1d - (high_1d - low_1d) * 1.1 / 6
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align 1d Camarilla levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1d trend filter: EMA34 on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 4h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        r2_val = r2_1d_aligned[i]
        s2_val = s2_1d_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R1 + above 1d EMA34 + volume spike
            if close[i] > r1_val and close[i] > ema34_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S1 + below 1d EMA34 + volume spike
            elif close[i] < s1_val and close[i] < ema34_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S1 or below 1d EMA34
            if close[i] < s1_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R1 or above 1d EMA34
            if close[i] > r1_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals