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
    
    # Daily high and low for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d / 2
    camarilla_s3 = close_1d - 1.1 * range_1d / 2
    
    # Align to 12h timeframe (previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Weekly trend: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 12h ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA34 for short-term momentum
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: volume > 1.5x 30-period SMA
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.5 * vol_ma30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(ema34[i]) or np.isnan(vol_ma30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3, above daily and weekly EMA34, with volume
            if (price > camarilla_r3_aligned[i] and 
                price > ema34_1d_aligned[i] and 
                price > ema34_1w_aligned[i] and 
                price > ema34[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S3, below daily and weekly EMA34, with volume
            elif (price < camarilla_s3_aligned[i] and 
                  price < ema34_1d_aligned[i] and 
                  price < ema34_1w_aligned[i] and 
                  price < ema34[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price crosses below EMA34 or loses volume
            if (price < ema34[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34 or loses volume
            if (price > ema34[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals