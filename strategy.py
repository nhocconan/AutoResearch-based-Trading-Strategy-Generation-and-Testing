#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for daily pivot levels and trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot levels (using prior day's OHLC)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    pivot = (high_d + low_d + close_d) / 3
    r1 = 2 * pivot - low_d
    s1 = 2 * pivot - high_d
    r2 = pivot + (high_d - low_d)
    s2 = pivot - (high_d - low_d)
    r3 = high_d + 2 * (pivot - low_d)
    s3 = low_d - 2 * (high_d - pivot)
    
    # Align daily pivots to 4h (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = prices['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation using 4h volume
    vol_4h = prices['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema50_4h[i]) or np.isnan(vol_ma_20_4h[i])):
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
        vol_current = prices['volume'].iloc[i]
        
        # Daily pivot levels
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        
        # Trend filter: price above/below 4h EMA50
        uptrend = price_close > ema50_4h[i]
        downtrend = price_close < ema50_4h[i]
        
        # Volume confirmation
        volume_confirm = vol_current > 1.5 * vol_ma_20_4h[i]
        
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
            # Fade at R1/S1 in ranging markets (price near extremes with rejection)
            elif (not uptrend and not downtrend and volume_confirm):
                # Fade at R1: price touches R1 and shows rejection (close < open)
                if abs(price_close - r1_val) < 0.003 * r1_val and price_close < prices['open'].iloc[i]:
                    signals[i] = -0.20
                    position = -1
                # Fade at S1: price touches S1 and shows rejection (close > open)
                elif abs(price_close - s1_val) < 0.003 * s1_val and price_close > prices['open'].iloc[i]:
                    signals[i] = 0.20
                    position = 1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below R1 OR stop loss at S1
                if (price_close < r1_val) or (price_close < s1_val):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above S1 OR stop loss at R1
                if (price_close > s1_val) or (price_close > r1_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DailyPivot_R1S1_R2S2_BreakoutFade"
timeframe = "4h"
leverage = 1.0