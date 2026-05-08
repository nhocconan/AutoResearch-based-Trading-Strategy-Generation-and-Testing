#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume_Session"
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
    
    # Get 4h data once for trend and volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 4h ATR(14) for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = np.append([close_4h[0]], close_4h[:-1])
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - close_4h_prev)
    tr3 = np.abs(low_4h - close_4h_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    
    # 1d close for Camarilla calculation (previous day)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.append([close_1d[0]], close_1d[:-1])
    
    # Pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d_prev) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, S3 (most significant)
    r3_1d = close_1d_prev + range_1d * 1.1 / 2
    s3_1d = close_1d_prev - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(atr14_4h_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema20_4h_aligned[i]
        atr_val = atr14_4h_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        vol_spike = volume_spike[i]
        in_session = session_filter[i]
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy markets)
        vol_filter = atr_val > (0.005 * close[i])
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above 4h EMA, in session, good volatility
            if (close[i] > r3_val and vol_spike and 
                close[i] > ema_val and in_session and vol_filter):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S3 with volume spike, below 4h EMA, in session, good volatility
            elif (close[i] < s3_val and vol_spike and 
                  close[i] < ema_val and in_session and vol_filter):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR below 4h EMA
            if (close[i] < s3_val or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 OR above 4h EMA
            if (close[i] > r3_val or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals