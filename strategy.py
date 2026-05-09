#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3_S3_Breakout_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate previous day's Camarilla R3 and S3 levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    if len(close_1d) >= 2:
        # Use previous day's OHLC for today's levels
        prev_high = high_1d[:-1]
        prev_low = low_1d[:-1]
        prev_close = close_1d[:-1]
        
        prev_range = prev_high - prev_low
        prev_pivot = (prev_high + prev_low + prev_close) / 3
        prev_r3 = prev_pivot + 1.1 * prev_range / 4
        prev_s3 = prev_pivot - 1.1 * prev_range / 4
        
        # Prepend NaN for first day
        camarilla_r3_1d = np.concatenate([[np.nan], prev_r3])
        camarilla_s3_1d = np.concatenate([[np.nan], prev_s3])
    else:
        camarilla_r3_1d = np.full_like(close_1d, np.nan)
        camarilla_s3_1d = np.full_like(close_1d, np.nan)
    
    # Align Camarilla levels and EMA to 1d
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or np.isnan(ema50_1d[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above Camarilla R3 with uptrend and volume spike
            if close[i] > r3_1d[i] and close[i] > ema50_1d[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 with downtrend and volume spike
            elif close[i] < s3_1d[i] and close[i] < ema50_1d[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Camarilla S3 OR trend turns down
            if close[i] < s3_1d[i] or close[i] < ema50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Camarilla R3 OR trend turns up
            if close[i] > r3_1d[i] or close[i] > ema50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals