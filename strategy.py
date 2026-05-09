#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3_S3_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (H1, L1, C1)
    H1 = df_1d['high'].values
    L1 = df_1d['low'].values
    C1 = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    R3 = C1 + (H1 - L1) * 1.1 / 4
    S3 = C1 - (H1 - L1) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 4h EMA34 for trend filter
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume spike detection
    vol_1d_series = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d volume spike: current 1d volume > 2.0 * 20-day MA of 1d volume
        vol_ok = df_1d['volume'].values[-1] > 2.0 * vol_ma20_1d_aligned[i] if len(df_1d) > 0 else False
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price breaks above R3 + above 4h EMA (uptrend) + volume spike + in session
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_4h_aligned[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 + below 4h EMA (downtrend) + volume spike + in session
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_4h_aligned[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or 4h EMA
            if close[i] < S3_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R3 or 4h EMA
            if close[i] > R3_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals