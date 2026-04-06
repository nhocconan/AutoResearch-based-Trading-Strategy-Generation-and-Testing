#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13995_6d_pivot_breakout_1w_dir_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot(high, low, close):
    """Calculate classic pivot points"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivots and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot(high_1d, low_1d, close_1d)
    
    # Align pivots to 6h timeframe (use previous 1d bar for pivot)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r3 + (r3 - s3))  # R4 = R3 + (R3-S3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s3 - (r3 - s3))  # S4 = S3 - (R3-S3)
    
    # Volume confirmation (20-period average on 1d)
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, 50)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h data for price and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(100, 50, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_aligned[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend bias from 1w EMA(50)
        bullish_trend = close[i] > ema_1w_aligned[i]  # price above 1w EMA50 = bullish
        bearish_trend = close[i] < ema_1w_aligned[i]  # price below 1w EMA50 = bearish
        
        # Volume confirmation (current 1d volume vs 20-day average)
        # We need current day's volume - approximate using 6h volume scaled
        # Simpler: use 6h volume vs its own 20-period average
        vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_ok = volume[i] > (vol_ma_6h[i] * 1.5)
        
        # Breakout signals at R4/S4 levels
        breakout_up = close[i] > r4_aligned[i]   # break above R4
        breakout_down = close[i] < s4_aligned[i] # break below S4
        
        # Fade signals at R3/S3 levels (only in ranging markets)
        # We'll use a simple range condition: price between R1 and S1
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        in_range = (close[i] > s1_aligned[i]) and (close[i] < r1_aligned[i])
        
        fade_up = in_range and (close[i] < r3_aligned[i]) and (close[i] > r3_aligned[i] - 0.1*(r3_aligned[i]-s3_aligned[i]))  # near R3
        fade_down = in_range and (close[i] > s3_aligned[i]) and (close[i] < s3_aligned[i] + 0.1*(r3_aligned[i]-s3_aligned[i]))  # near S3
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        if bullish_trend and volume_ok and breakout_up:
            long_signal = True
        elif bearish_trend and volume_ok and breakout_down:
            short_signal = True
        elif not bullish_trend and not bearish_trend:  # ranging market
            if volume_ok and fade_down:
                long_signal = True  # buy near S3
            elif volume_ok and fade_up:
                short_signal = True  # sell near R3
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown below S3 or trend change to bearish
            if close[i] < s3_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on breakout above R3 or trend change to bullish
            if close[i] > r3_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals