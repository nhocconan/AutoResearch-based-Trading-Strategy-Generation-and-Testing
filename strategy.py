#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter and weekly context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 0.55
    s3 = prev_close - range_hl * 0.55
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: 20-period average volume for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR > 0.3% of price
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # Ensure volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period average
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3, above 1w EMA20 (uptrend), with volume spike
            buffer = 0.001 * close[i]
            if (close[i] > r3_aligned[i] + buffer and 
                close[i] > ema_20_1w_aligned[i] + buffer and
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, below 1w EMA20 (downtrend), with volume spike
            elif (close[i] < s3_aligned[i] - buffer and 
                  close[i] < ema_20_1w_aligned[i] - buffer and
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to midpoint of prior day's range (H3/L3)
            range_hl = prev_high - prev_low
            h3 = prev_close + range_hl * 0.275
            l3 = prev_close - range_hl * 0.275
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
            
            camarilla_mid = (h3_aligned[i] + l3_aligned[i]) / 2
            range_hl_1d = h3_aligned[i] - l3_aligned[i]
            at_mid = abs(close[i] - camarilla_mid) < range_hl_1d * 0.30
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals