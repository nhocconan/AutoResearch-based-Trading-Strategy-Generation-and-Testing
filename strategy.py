# MERGED STRATEGY v1 - Combined Best Elements
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Donchian_Trend_Filter"
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
    
    # 1d data for Camarilla and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    n1d = len(close_1d)
    camarilla_S1 = np.full(n1d, np.nan)
    camarilla_S2 = np.full(n1d, np.nan)
    camarilla_R1 = np.full(n1d, np.nan)
    camarilla_R2 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_val = H - L
        camarilla_S1[i] = C - range_val * 1.08
        camarilla_S2[i] = C - range_val * 1.16
        camarilla_R1[i] = C + range_val * 1.08
        camarilla_R2[i] = C + range_val * 1.16
    
    # Calculate Donchian channels from previous day
    donchian_lower = np.full(n1d, np.nan)
    donchian_upper = np.full(n1d, np.nan)
    for i in range(1, n1d):
        donchian_lower[i] = np.min(low_1d[:i])
        donchian_upper[i] = np.max(high_1d[:i])
    
    # Align to 4h timeframe
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_S2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S2)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_R2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R2)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    
    # 12h data for trend filter (stronger filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume spike: current > 1.5x 20-period average (more sensitive)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_S2_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_R2_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above BOTH Camarilla S1 AND Donchian lower + uptrend + volume
            long_cond = (close[i] > camarilla_S1_aligned[i] and 
                        close[i] > donchian_lower_aligned[i] and
                        ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price below BOTH Camarilla R1 AND Donchian upper + downtrend + volume
            short_cond = (close[i] < camarilla_R1_aligned[i] and 
                         close[i] < donchian_upper_aligned[i] and
                         ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below Camarilla S2 OR Donchian lower
            if close[i] < camarilla_S2_aligned[i] or close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above Camarilla R2 OR Donchian upper
            if close[i] > camarilla_R2_aligned[i] or close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals