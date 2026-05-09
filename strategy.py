#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # Daily EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Camarilla levels from previous 1d bar
    # R3 = close + 1.1 * (high - low) / 6
    # S3 = close - 1.1 * (high - low) / 6
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    camarilla_R3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    camarilla_S3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume spike: current > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or \
           np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume spike
            if (price > camarilla_R3_aligned[i] and 
                price > ema34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S3 + downtrend + volume spike
            elif (price < camarilla_S3_aligned[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price closes below EMA34 or loses momentum
            if price < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above EMA34 or loses momentum
            if price > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals