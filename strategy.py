#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels (R3/S3) for breakout entries, 
# with 1d volume confirmation and 1w ADX trend filter. Designed for low trade frequency 
# (target 20-50 trades/year) to avoid fee drag. Uses daily structure for key levels, 
# daily volume surge for momentum confirmation, and weekly ADX to ensure we only trade 
# in strong trends. Works in bull/bear markets by following higher timeframe trends 
# with strict entry filters based on institutional pivot levels.

name = "12h_Camarilla_R3S3_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (using previous day's data)
    # Classic formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We'll use R3 and S3 for entries
    def calculate_camarilla(high, low, close):
        # Calculate for each day using previous day's data
        pivot = (high + low + close) / 3.0
        range_ = high - low
        
        r3 = close + range_ * 1.1 / 4.0
        s3 = close - range_ * 1.1 / 4.0
        
        return r3, s3
    
    # Calculate Camarilla levels using previous day's data (to avoid look-ahead)
    r3_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        r3_1d[i], s3_1d[i] = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        def WilderMA(arr, period):
            res = np.full_like(arr, np.nan)
            if len(arr) < period:
                return res
            # First value is simple average
            res[period-1] = np.nanmean(arr[1:period])
            # Rest is Wilder smoothing
            for i in range(period, len(arr)):
                res[i] = (res[i-1] * (period-1) + arr[i]) / period
            return res
        
        tr_ma = WilderMA(tr, period)
        plus_dm_ma = WilderMA(plus_dm, period)
        minus_dm_ma = WilderMA(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_ma != 0, 100 * plus_dm_ma / tr_ma, 0)
        minus_di = np.where(tr_ma != 0, 100 * minus_dm_ma / tr_ma, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = WilderMA(dx, period)
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Volume spike: 2x 20-day EMA
    vol_ema = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ema * 2.0)
    
    # Align all indicators to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when ADX > 25 (strong trend)
        if adx_1w_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above daily R3 + volume surge
            if close[i] > r3_1d_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily S3 + volume surge
            elif close[i] < s3_1d_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily S3 or ADX weakens
            if close[i] < s3_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily R3 or ADX weakens
            if close[i] > r3_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals