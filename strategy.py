#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_volume_trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    H4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    L4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter - 30-period average on 12h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend from weekly EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Camarilla breakout with volume confirmation
        # Long: Break above H4 in uptrend
        long_signal = close[i] > H4_aligned[i] and uptrend and volume_ok[i]
        # Short: Break below L4 in downtrend
        short_signal = close[i] < L4_aligned[i] and downtrend and volume_ok[i]
        
        # Exit when price returns to mid-point (pivot)
        pivot = (H4_aligned[i] + L4_aligned[i]) / 2
        exit_long = close[i] < pivot
        exit_short = close[i] > pivot
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals