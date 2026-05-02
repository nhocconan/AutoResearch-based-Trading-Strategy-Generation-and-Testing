#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1w EMA50 trend + volume spike
# Camarilla pivot levels (R3/S3) act as strong support/resistance from 1d data.
# Breakout above R3 or below S3 with volume confirmation captures institutional moves.
# 1w EMA50 filter ensures alignment with weekly trend, working in both bull and bear markets.
# Volume spike (>2x 20-period average) confirms participation.
# Target: 75-200 trades over 4 years (19-50/year) on 4h.

name = "4h_Camarilla_R3S3_Breakout_1wEMA50_Volume"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4.0
    r3_1d = close_1d + camarilla_range
    s3_1d = close_1d - camarilla_range
    
    # AlCamarilla levels to 4h timeframe (use previous day's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation: 2.0x 20-period average
    if len(volume) < 20:
        return np.zeros(n)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA and volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with volume spike AND price > 1w EMA50 (bullish trend)
            if (close[i] > r3_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 with volume spike AND price < 1w EMA50 (bearish trend)
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla pivot point (mid-level) OR below 1w EMA50 (trend change)
            # Camarilla pivot point = (H+L+C)/3 from 1d data
            pp_1d = (high_1d + low_1d + close_1d) / 3.0
            pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
            if (np.isnan(pp_1d_aligned[i]) or 
                close[i] < pp_1d_aligned[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla pivot point OR above 1w EMA50 (trend change)
            pp_1d = (high_1d + low_1d + close_1d) / 3.0
            pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
            if (np.isnan(pp_1d_aligned[i]) or 
                close[i] > pp_1d_aligned[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals