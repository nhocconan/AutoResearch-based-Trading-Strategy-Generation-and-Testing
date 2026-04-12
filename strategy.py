#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
    # Camarilla levels (R3/S3, R4/S4) act as institutional support/resistance on 1d
    # 4h EMA(21) trend filter ensures we only trade in direction of higher timeframe trend
    # Session filter (08-20 UTC) avoids low volume Asian session
    # Breakout above R4 or below S4 with volume confirmation = continuation signal
    # Target: 15-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # R3 = C + (Range * 1.1/4)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 1h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA(21) for trend
    close_4h_series = pd.Series(close_4h)
    ema_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Volume confirmation: current 1h volume > 1.5x 4h average volume per hour
        # Approximate 4h average volume per hour: volume_4h / 4
        vol_ma_4h = np.full(len(df_4h), np.nan)
        for j in range(20, len(df_4h)):
            if j == 20:
                vol_ma_4h[j] = np.mean(volume_1d[j*4:(j+1)*4]) if (j+1)*4 <= len(volume_1d) else np.nan
            else:
                # Simple approximation: use 4h volume MA
                start_idx = max(0, j*4 - 20)
                end_idx = min(len(volume), j*4)
                if end_idx > start_idx:
                    vol_ma_4h[j] = np.mean(volume[start_idx:end_idx])
                else:
                    vol_ma_4h[j] = vol_ma_4h[j-1] if j > 0 else np.nan
        # Simplified volume check: compare to rolling mean of volume
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i+1])
            vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 1.0
        else:
            vol_ratio = 1.0
        
        # Breakout signals: price breaks R4/S4 with volume expansion AND trend alignment
        breakout_long = (close[i] > r4_aligned[i]) and (vol_ratio > 1.5) and uptrend
        breakout_short = (close[i] < s4_aligned[i]) and (vol_ratio > 1.5) and downtrend
        
        # Exit conditions: return to pivot or opposite extreme
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_1d_4h_camarilla_breakout_trend_v1"
timeframe = "1h"
leverage = 1.0