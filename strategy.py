#!/usr/bin/env python3
name = "6H_Camarilla_R3_S3_12HTrend_VolumeSpike_v1"
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
    
    # Get 6h data for structure (R3/S3 levels)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate previous 6h bar's high, low, close for Camarilla levels
    prev_high = df_6h['high'].values
    prev_low = df_6h['low'].values
    prev_close = df_6h['close'].values
    
    # Calculate Camarilla levels: R3 and S3 (correct formula)
    r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR > 0.3% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    # Session filter: 08:00 - 20:00 UTC (80% of day)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + 12h uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and   # 12h uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + 12h downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and   # 12h downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the middle of the prior 6h range (H4/L4)
            # H4 = close + 1.1*(high-low)*1.1/6, L4 = close - 1.1*(high-low)*1.1/6
            h4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 6
            l4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 6
            h4_aligned = align_htf_to_ltf(prices, df_6h, h4)
            l4_aligned = align_htf_to_ltf(prices, df_6h, l4)
            
            camarilla_mid = (h4_aligned[i] + l4_aligned[i]) / 2
            at_mid = abs(close[i] - camarilla_mid) < (h4_aligned[i] - l4_aligned[i]) * 0.25  # Within 25% of range
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals