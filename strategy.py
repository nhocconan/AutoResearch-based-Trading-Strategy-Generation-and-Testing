#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Camarilla levels (R3, S3) - calculated from previous day's range
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # Using previous day's values (shifted by 1) to avoid look-ahead
    daily_high = pd.Series(high).shift(1)
    daily_low = pd.Series(low).shift(1)
    daily_close = pd.Series(close).shift(1)
    
    R3 = daily_close + 1.1 * (daily_high - daily_low) / 2
    S3 = daily_close - 1.1 * (daily_high - daily_low) / 2
    
    # Align daily levels to 12h timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily data
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    d_R3 = d_close + 1.1 * (d_high - d_low) / 2
    d_S3 = d_close - 1.1 * (d_high - d_low) / 2
    
    # Shift to use previous day's levels (avoid look-ahead)
    d_R3 = np.roll(d_R3, 1)
    d_S3 = np.roll(d_S3, 1)
    d_R3[0] = np.nan
    d_S3[0] = np.nan
    
    R3_aligned = align_htf_to_ltf(prices, df_1d, d_R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, d_S3)
    
    # Daily trend filter: EMA34
    d_ema34 = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    d_ema34_aligned = align_htf_to_ltf(prices, df_1d, d_ema34)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(d_ema34_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 + daily uptrend + volume confirmation
            if (price > R3_aligned[i] and 
                price > d_ema34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S3 + daily downtrend + volume confirmation
            elif (price < S3_aligned[i] and 
                  price < d_ema34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns below EMA34 or volume drops
            if (price < d_ema34_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above EMA34 or volume drops
            if (price > d_ema34_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals