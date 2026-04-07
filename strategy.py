#!/usr/bin/env python3
"""
1h Williams %R + 4h ADX Trend Filter + 1d Session Filter
Long when Williams %R crosses above -80 in bullish 4h trend (ADX>25) and 1d EMA50>EMA200
Short when Williams %R crosses below -20 in bearish 4h trend (ADX>25) and 1d EMA50<EMA200
Exit when Williams %R crosses back through -50
Uses 4h for trend direction, 1h for entry timing, and 1d session filter to avoid low-volume hours
Target: 15-35 trades/year per symbol (60-140 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_williams_r_4h_adx_1d_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Williams %R (14) on 1h ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    diff = highest_high - lowest_low
    diff = np.where(diff == 0, 1, diff)  # Avoid division by zero
    willr = -100 * (highest_high - close) / diff  # Williams %R
    
    # === 4h ADX Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- (14 periods)
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 4h ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # === 1d EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === 1d Session Filter (8-20 UTC) ===
    # Pre-compute hour array once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter (8-20 UTC)
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50
            if willr[i] < -50 and willr[i-1] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50
            if willr[i] > -50 and willr[i-1] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Bullish conditions: ADX>25 (trending) + EMA50>EMA200 (uptrend)
            bullish_trend = adx_aligned[i] > 25 and ema_50_aligned[i] > ema_200_aligned[i]
            # Bearish conditions: ADX>25 (trending) + EMA50<EMA200 (downtrend)
            bearish_trend = adx_aligned[i] > 25 and ema_50_aligned[i] < ema_200_aligned[i]
            
            if bullish_trend:
                # Look for long entry: Williams %R crosses above -80 (oversold bounce)
                if willr[i] > -80 and willr[i-1] <= -80:
                    position = 1
                    signals[i] = 0.20
            elif bearish_trend:
                # Look for short entry: Williams %R crosses below -20 (overbought rejection)
                if willr[i] < -20 and willr[i-1] >= -20:
                    position = -1
                    signals[i] = -0.20
    
    return signals