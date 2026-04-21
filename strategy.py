#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for 12h pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Load 1h data for trend filter (12h EMA)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 12h pivot levels (using prior 12h bar's OHLC)
    high_12h = df_1d['high'].values
    low_12d = df_1d['low'].values
    close_12h = df_1d['close'].values
    
    pivot_12h = (high_12h + low_12d + close_12h) / 3
    r1_12h = 2 * pivot_12h - low_12d
    s1_12h = 2 * pivot_12h - high_12h
    r2_12h = pivot_12h + (high_12h - low_12d)
    s2_12h = pivot_12h - (high_12h - low_12d)
    r3_12h = high_12h + 2 * (pivot_12h - low_12d)
    s3_12h = low_12d - 2 * (high_12h - pivot_12h)
    r4_12h = r3_12h + (high_12h - low_12d)
    s4_12h = s3_12h - (high_12h - low_12d)
    
    # Align 12h pivots to 6h (wait for 12h close)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_1d, pivot_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_1d, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_1d, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_1d, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_1d, s4_12h)
    
    # Calculate 1h EMA50 for trend filter
    close_1h = df_1h['close'].values
    ema50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    # Volume confirmation using 6h volume
    vol_6h = prices['volume'].values
    vol_ma_20_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema50_1h_aligned[i]) or np.isnan(vol_ma_20_6h[i])):
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
        vol_current = vol_6h[i]
        
        # 12h pivot levels
        r3_val = r3_12h_aligned[i]
        s3_val = s3_12h_aligned[i]
        r4_val = r4_12h_aligned[i]
        s4_val = s4_12h_aligned[i]
        
        # Trend filter: price above/below 1h EMA50
        uptrend = price_close > ema50_1h_aligned[i]
        downtrend = price_close < ema50_1h_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_current > 1.5 * vol_ma_20_6h[i]
        
        if position == 0:
            # Enter long: price breaks above R4 with volume in uptrend
            if (uptrend and 
                price_close > r4_val and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S4 with volume in downtrend
            elif (downtrend and 
                  price_close < s4_val and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            # Fade at R3/S3 in ranging markets (price near extremes with rejection)
            elif (not uptrend and not downtrend and volume_confirm):
                # Fade at R3: price touches R3 and shows rejection (close < open)
                if abs(price_close - r3_val) < 0.005 * r3_val and price_close < prices['open'].iloc[i]:
                    signals[i] = -0.20
                    position = -1
                # Fade at S3: price touches S3 and shows rejection (close > open)
                elif abs(price_close - s3_val) < 0.005 * s3_val and price_close > prices['open'].iloc[i]:
                    signals[i] = 0.20
                    position = 1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below R3 OR stop loss at S3
                if (price_close < r3_val) or (price_close < s3_val):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above S3 OR stop loss at R3
                if (price_close > s3_val) or (price_close > r3_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_12hPivot_R3S3_R4S4_BreakoutFade"
timeframe = "6h"
leverage = 1.0