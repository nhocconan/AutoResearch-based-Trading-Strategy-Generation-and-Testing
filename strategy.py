#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels (primary levels)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (standard formula)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Standard Camarilla: P = (H + L + C)/3
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    # Resistance/Support levels (Camarilla specific)
    r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12
    s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12
    r2_4h = close_4h + (high_4h - low_4h) * 1.1 / 6
    s2_4h = close_4h - (high_4h - low_4h) * 1.1 / 6
    r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4
    s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4
    r4_4h = close_4h + (high_4h - low_4h) * 1.1 / 2
    s4_4h = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # Align 4h Camarilla levels to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # Get 4h trend filter: EMA50 on 4h close
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d volume filter: current 1d volume > 2.0 * 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_series_1d = pd.Series(volume_1d)
    vol_ma_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike_1d = volume_1d > (vol_ma_1d_aligned * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(r2_4h_aligned[i]) or
            np.isnan(s2_4h_aligned[i]) or
            np.isnan(r3_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or
            np.isnan(r4_4h_aligned[i]) or
            np.isnan(s4_4h_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_4h_aligned[i]
        r1_val = r1_4h_aligned[i]
        s1_val = s1_4h_aligned[i]
        r2_val = r2_4h_aligned[i]
        s2_val = s2_4h_aligned[i]
        r3_val = r3_4h_aligned[i]
        s3_val = s3_4h_aligned[i]
        r4_val = r4_4h_aligned[i]
        s4_val = s4_4h_aligned[i]
        ema50_val = ema50_4h_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        in_session = session_filter[i]
        
        if position == 0:
            # Enter long: Price above R1 + above 4h EMA50 + volume spike + in session
            if close[i] > r1_val and close[i] > ema50_val and vol_spike and in_session:
                signals[i] = 0.20
                position = 1
            # Enter short: Price below S1 + below 4h EMA50 + volume spike + in session
            elif close[i] < s1_val and close[i] < ema50_val and vol_spike and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S1 or below 4h EMA50
            if close[i] < s1_val or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price rises above R1 or above 4h EMA50
            if close[i] > r1_val or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals