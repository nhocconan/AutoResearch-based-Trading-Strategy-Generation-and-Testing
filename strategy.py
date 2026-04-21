#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for pivot levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot levels (using prior day's OHLC)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    vol_d = df_1d['volume'].values
    
    pivot_d = (high_d + low_d + close_d) / 3
    r1_d = 2 * pivot_d - low_d
    s1_d = 2 * pivot_d - high_d
    r2_d = pivot_d + (high_d - low_d)
    s2_d = pivot_d - (high_d - low_d)
    
    # Align daily data to 4h (wait for daily close)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    r2_d_aligned = align_htf_to_ltf(prices, df_1d, r2_d)
    s2_d_aligned = align_htf_to_ltf(prices, df_1d, s2_d)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation using 1d volume
    vol_ma_20_1d = pd.Series(vol_d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_d_aligned[i]) or np.isnan(r1_d_aligned[i]) or np.isnan(s1_d_aligned[i]) or
            np.isnan(r2_d_aligned[i]) or np.isnan(s2_d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        vol_current = align_htf_to_ltf(prices, df_1d, vol_d)[i]  # 1d volume aligned to 4h
        
        # Daily pivot levels
        r1_val = r1_d_aligned[i]
        s1_val = s1_d_aligned[i]
        r2_val = r2_d_aligned[i]
        s2_val = s2_d_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price_close > ema50_1d_aligned[i]
        downtrend = price_close < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_current > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R2 with volume in uptrend
            if (uptrend and 
                price_close > r2_val and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 with volume in downtrend
            elif (downtrend and 
                  price_close < s2_val and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below R1
                if price_close < r1_val:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above S1
                if price_close > s1_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DailyPivot_R1S1_R2S2_Breakout"
timeframe = "4h"
leverage = 1.0