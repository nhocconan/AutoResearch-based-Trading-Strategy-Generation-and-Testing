#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Load 1d data for trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using prior week's OHLC)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    r3 = high_w + 2 * (pivot - low_w)
    s3 = low_w - 2 * (high_w - pivot)
    r4 = r3 + (high_w - low_w)
    s4 = s3 - (high_w - low_w)
    
    # Align weekly pivots to 6h (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation using 6h volume
    vol_6h = df_1d['volume'].values  # Use 1d volume as proxy for 6h volume confirmation
    vol_ma_20_1d = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
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
        vol_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        # Weekly pivot levels
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price_close > ema50_1d_aligned[i]
        downtrend = price_close < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_current > 1.3 * vol_ma_20_1d_aligned[i]
        
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

name = "6h_WeeklyPivot_R3S3_R4S4_BreakoutFade"
timeframe = "6h"
leverage = 1.0