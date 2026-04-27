#!/usr/bin/env python3
"""
Strategy: 1d_Camarilla_H4_Trend_Volume
Hypothesis: Daily Camarilla pivot (R3/S3) breakout with 4-hour trend filter and volume confirmation.
- Uses daily Camarilla levels calculated from prior day OHLC for structure.
- Filters breakouts with 4h EMA trend (EMA20 > EMA50 for long, EMA20 < EMA50 for short).
- Requires volume > 1.5x 20-period average to avoid false breakouts.
- Works in bull/bear: Camarilla adapts to daily range; 4h trend filter avoids counter-trend entries.
- Designed for 1d timeframe: expects ~10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (prior day OHLC)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day OHLC
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Shift by 1 to use prior day's levels
    high_d_prev = np.concatenate([[np.nan], high_d[:-1]])
    low_d_prev = np.concatenate([[np.nan], low_d[:-1]])
    close_d_prev = np.concatenate([[np.nan], close_d[:-1]])
    
    rang = high_d_prev - low_d_prev
    camarilla_r3 = close_d_prev + rang * 1.1 / 4
    camarilla_s3 = close_d_prev - rang * 1.1 / 4
    
    # Align daily Camarilla to 1d (no extra delay for prior day's levels)
    r3_d_aligned = align_htf_to_ltf(prices, df_d, camarilla_r3)
    s3_d_aligned = align_htf_to_ltf(prices, df_d, camarilla_s3)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMAs
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMAs to 1d
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(vol_period, 50) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(r3_d_aligned[i]) or np.isnan(s3_d_aligned[i]) or
            np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above R3 with volume AND 4h uptrend (EMA20 > EMA50)
            if price > r3_d_aligned[i] and vol_ratio > 1.5 and ema20_4h_aligned[i] > ema50_4h_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below S3 with volume AND 4h downtrend (EMA20 < EMA50)
            elif price < s3_d_aligned[i] and vol_ratio > 1.5 and ema20_4h_aligned[i] < ema50_4h_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below S3 or 4h trend turns down
            if price < s3_d_aligned[i] or ema20_4h_aligned[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above R3 or 4h trend turns up
            if price > r3_d_aligned[i] or ema20_4h_aligned[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_H4_Trend_Volume"
timeframe = "1d"
leverage = 1.0