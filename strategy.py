#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF structure with Camarilla pivot breakout and volume confirmation.
# Uses 4h Camarilla levels (R3/S3) for breakout direction and 1d EMA50 for trend filter.
# Long when: price breaks above 4h R3 with volume spike AND price > 1d EMA50.
# Short when: price breaks below 4h S3 with volume spike AND price < 1d EMA50.
# Uses session filter (08-20 UTC) to reduce noise. Fixed size 0.20.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
# Camarilla levels identify key intraday support/resistance; volume confirms breakout validity.
# 1d EMA50 ensures trades align with higher timeframe trend, reducing counter-trend whipsaws.
# Works in bull markets (trend-following breakouts) and bear markets (fade false breakouts at extremes).

name = "1h_Camarilla_R3S3_Breakout_4hVol_1dEMA50_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot points (using prior 4h bar OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Prior 4h bar OHLC for current 4h bar's Camarilla levels
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_4h = (prev_high + prev_low + prev_close) / 3.0
    range_4h = prev_high - prev_low
    r3_4h = prev_close + (range_4h * 1.1 / 4)
    s3_4h = prev_close - (range_4h * 1.1 / 4)
    r1_4h = prev_close + (range_4h * 1.1 / 12)
    s1_4h = prev_close - (range_4h * 1.1 / 12)
    
    # Align 4h Camarilla levels to 1h
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detector: volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for volume MA and HTF alignment
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_4h_aligned[i]
        curr_s3 = s3_4h_aligned[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 4h R3 with volume spike AND price > 1d EMA50
            if (curr_high > curr_r3 and 
                curr_vol_spike and 
                curr_close > curr_ema50):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h S3 with volume spike AND price < 1d EMA50
            elif (curr_low < curr_s3 and 
                  curr_vol_spike and 
                  curr_close < curr_ema50):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below 4h S1 OR price < 1d EMA50
            if (curr_low < s1_4h_aligned[i] or 
                curr_close < curr_ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above 4h R1 OR price > 1d EMA50
            if (curr_high > r1_4h_aligned[i] or 
                curr_close > curr_ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals