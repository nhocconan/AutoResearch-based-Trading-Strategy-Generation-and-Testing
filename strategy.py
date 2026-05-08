#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMMA) and Jaw > Teeth > Lips (bullish alignment)
# Short when price < Alligator Jaw and Jaw < Teeth < Lips (bearish alignment)
# Uses 1d EMA50 for trend filter and volume > 1.5x 20-period EMA of volume for confirmation
# Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years)
# Williams Alligator identifies trend direction and alignment, reducing whipsaw in sideways markets

name = "4h_Williams_Alligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # First value is SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: (prev*(period-1) + current) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    # Williams Alligator on 4h timeframe (13,8,5 periods)
    # Jaw: 13-period SMMA of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward  
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply forward shifts (Williams Alligator specifics)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    # Jaw shifted 8 bars
    for i in range(8, len(jaw_raw)):
        if not np.isnan(jaw_raw[i]):
            jaw[i+8] = jaw_raw[i]
            
    # Teeth shifted 5 bars  
    for i in range(5, len(teeth_raw)):
        if not np.isnan(teeth_raw[i]):
            teeth[i+5] = teeth_raw[i]
            
    # Lips shifted 3 bars
    for i in range(3, len(lips_raw)):
        if not np.isnan(lips_raw[i]):
            lips[i+3] = lips_raw[i]
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # warmup for Alligator (max shift 8 + jaw period 13)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period EMA
        # Find the most recent completed 1d bar
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed 1d bar
        
        if idx_1d < 0:
            vol_filter = False
        else:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            vol_filter = vol_1d_current > 1.5 * vol_ema_20_aligned[i]
        
        # Williams Alligator conditions
        # Bullish: price > jaw AND jaw > teeth > lips
        bullish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        bearish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        
        if position == 0:
            # Look for entry: Alligator alignment + trend + volume
            long_condition = (close[i] > jaw[i]) and bullish_alignment and (close[i] > ema_50_aligned[i]) and vol_filter
            short_condition = (close[i] < jaw[i]) and bearish_alignment and (close[i] < ema_50_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment or price crosses below jaw
            if bearish_alignment or (close[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment or price crosses above jaw
            if bullish_alignment or (close[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals