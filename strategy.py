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
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for additional context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly high/low/close for calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Daily high/low/close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly range for pivot calculations
    weekly_range = high_1w - low_1w
    
    # Weekly Camarilla pivot levels (based on previous week)
    camarilla_r4 = close_1w + weekly_range * 1.1 / 2
    camarilla_s4 = close_1w - weekly_range * 1.1 / 2
    
    # Align Weekly Camarilla levels to 1d timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Daily trend filter: EMA21
    close_1d_series = pd.Series(close_1d)
    ema21_1d = close_1d_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume filter: above average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA21
        trend_up = close[i] > ema21_1d_aligned[i]
        trend_down = close[i] < ema21_1d_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above weekly R4 with volume and trend up
        # Short: price breaks below weekly S4 with volume and trend down
        long_entry = (close[i] > r4_aligned[i]) and vol_filter and trend_up
        short_entry = (close[i] < s4_aligned[i]) and vol_filter and trend_down
        
        # Exit conditions: price returns to opposite weekly S4/R4 levels
        long_exit = (close[i] < s4_aligned[i])
        short_exit = (close[i] > r4_aligned[i])
        
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

name = "1d_WeeklyCamarilla_R4S4_DailyTrend_Volume"
timeframe = "1d"
leverage = 1.0