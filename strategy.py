#!/usr/bin/env python3
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily high/low/close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points (Camarilla formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_ = high_1d - low_1d
    r3 = close_1d + range_ * 1.1 / 4
    s3 = close_1d - range_ * 1.1 / 4
    r4 = close_1d + range_ * 1.1 / 2
    s4 = close_1d - range_ * 1.1 / 2
    
    # Align pivot levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Weekly trend filter: price above/below weekly EMA20
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 0-23 UTC (trade all day for daily timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: trade all day for daily timeframe
        hour = hours[i]
        in_session = True  # Trade all hours for daily timeframe
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above R4 with volume and trend up
        # Short: price breaks below S4 with volume and trend down
        long_entry = (close[i] > r4_aligned[i]) and vol_filter and trend_up
        short_entry = (close[i] < s4_aligned[i]) and vol_filter and trend_down
        
        # Exit conditions: price returns to opposite S3/R3 levels
        long_exit = (close[i] < s3_aligned[i])
        short_exit = (close[i] > r3_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R4_S4_WeeklyEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0