#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla levels (R3, S3) for 12h breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    prev_close_1d = df_1d['close'].shift(1).values  # previous day close
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Camarilla R3 and S3 levels
    R3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 6
    S3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 6
    
    # Align daily levels to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily trend filter: EMA34 on 1d
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period SMA of volume
    vol_sma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > 1.5 * vol_sma24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or \
           np.isnan(ema34_12h[i]) or np.isnan(vol_sma24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 + daily uptrend + volume spike
            if (price > R3_12h[i] and 
                price > ema34_12h[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S3 + daily downtrend + volume spike
            elif (price < S3_12h[i] and 
                  price < ema34_12h[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns below EMA34 or volume drops
            if (price < ema34_12h[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above EMA34 or volume drops
            if (price > ema34_12h[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals