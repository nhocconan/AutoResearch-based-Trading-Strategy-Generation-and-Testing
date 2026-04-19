#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Volume_Session_v2"
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
    
    # Get 4h and 1d data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h ATR(14) for Camarilla width
    tr4h = np.maximum(df_4h['high'][1:], df_4h['close'][:-1]) - np.minimum(df_4h['low'][1:], df_4h['close'][:-1])
    tr4h = np.maximum(tr4h, np.abs(df_4h['high'][1:] - df_4h['close'][:-1]))
    tr4h = np.maximum(tr4h, np.abs(df_4h['low'][1:] - df_4h['close'][:-1]))
    tr4h = np.concatenate([[np.nan], tr4h])
    atr_14_4h = pd.Series(tr4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR(14) for Camarilla width
    tr1d = np.maximum(df_1d['high'][1:], df_1d['close'][:-1]) - np.minimum(df_1d['low'][1:], df_1d['close'][:-1])
    tr1d = np.maximum(tr1d, np.abs(df_1d['high'][1:] - df_1d['close'][:-1]))
    tr1d = np.maximum(tr1d, np.abs(df_1d['low'][1:] - df_1d['close'][:-1]))
    tr1d = np.concatenate([[np.nan], tr1d])
    atr_14_1d = pd.Series(tr1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Camarilla levels using previous 4h bar's data
    prev_close_4h = np.concatenate([[np.nan], df_4h['close'][:-1]])
    prev_high_4h = np.concatenate([[np.nan], df_4h['high'][:-1]])
    prev_low_4h = np.concatenate([[np.nan], df_4h['low'][:-1]])
    
    camarilla_H4_4h = prev_close_4h + 1.1/2 * (prev_high_4h - prev_low_4h)
    camarilla_L4_4h = prev_close_4h - 1.1/2 * (prev_high_4h - prev_low_4h)
    
    # Calculate 1d Camarilla levels using previous day's data
    prev_close_1d = np.concatenate([[np.nan], df_1d['close'][:-1]])
    prev_high_1d = np.concatenate([[np.nan], df_1d['high'][:-1]])
    prev_low_1d = np.concatenate([[np.nan], df_1d['low'][:-1]])
    
    camarilla_H4_1d = prev_close_1d + 1.1/2 * (prev_high_1d - prev_low_1d)
    camarilla_L4_1d = prev_close_1d - 1.1/2 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_H4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H4_4h)
    camarilla_L4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L4_4h)
    camarilla_H4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4_1d)
    camarilla_L4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4_1d)
    
    # Volume filter: current volume > 2.0x 24-period average (more stringent)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_H4_4h_aligned[i]) or np.isnan(camarilla_L4_4h_aligned[i]) or \
           np.isnan(camarilla_H4_1d_aligned[i]) or np.isnan(camarilla_L4_1d_aligned[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        hour = hours[i]
        
        # Session filter: 08-20 UTC
        in_session = (8 <= hour <= 20)
        
        # Volume filter
        volume_ok = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above both 4h and 1d H4 with volume and in session
            if price > camarilla_H4_4h_aligned[i] and price > camarilla_H4_1d_aligned[i] and volume_ok and in_session:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below both 4h and 1d L4 with volume and in session
            elif price < camarilla_L4_4h_aligned[i] and price < camarilla_L4_1d_aligned[i] and volume_ok and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price returns below either 4h or 1d H4
            if price < camarilla_H4_4h_aligned[i] or price < camarilla_H4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price returns above either 4h or 1d L4
            if price > camarilla_L4_4h_aligned[i] or price > camarilla_L4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals