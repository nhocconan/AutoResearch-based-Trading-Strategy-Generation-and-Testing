#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(10) for volatility filter
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate previous day's close for Camarilla pivot calculation
    prev_close_1d = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high_1d = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low_1d = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    # Calculate Camarilla levels for each day (R4/S4 for breakout, R3/S3 for fade)
    R4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    R3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 4
    S3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 4
    S4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align daily indicators to 12h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate 12-period moving average of volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Volatility filter: only trade when ATR is above its 20-period average
        atr_ma = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean()
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma.values)
        vol_filter = atr_aligned[i] > atr_ma_aligned[i] if not np.isnan(atr_ma_aligned[i]) else False
        
        # Volume filter: current volume above average
        vol_filter = vol_filter and volume[i] > vol_ma[i]
        
        # Breakout conditions: price breaks R4 (long) or S4 (short) with volume
        long_breakout = close[i] > R4_aligned[i] and vol_filter
        short_breakout = close[i] < S4_aligned[i] and vol_filter
        
        # Fade conditions: price touches R3/S3 and reverses
        long_fade = (close[i] <= R3_aligned[i] * 1.002) and (close[i] >= R3_aligned[i] * 0.998) and downtrend and vol_filter
        short_fade = (close[i] >= S3_aligned[i] * 0.998) and (close[i] <= S3_aligned[i] * 1.002) and uptrend and vol_filter
        
        if (long_breakout or long_fade) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_breakout or short_fade) and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite signal or trend reversal
        elif position == 1 and (short_breakout or short_fade or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (long_breakout or long_fade or not downtrend):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3S4_BreakoutFade_VolumeTrend"
timeframe = "12h"
leverage = 1.0