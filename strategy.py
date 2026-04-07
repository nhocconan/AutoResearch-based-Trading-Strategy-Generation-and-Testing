#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use Camarilla pivot levels from weekly timeframe for support/resistance, with volume confirmation and trend filter from daily timeframe.
Enter long when price touches S3 level with bullish daily trend and volume > 1.5x average.
Enter short when price touches R3 level with bearish daily trend and volume > 1.5x average.
Exit when price moves to opposite pivot level or trend reverses.
Targets 12-37 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Camarilla pivots (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from weekly OHLC
    # Camarilla: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4
    #          R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot levels
    H_minus_L = weekly_high - weekly_low
    camarilla_S3 = weekly_close - (H_minus_L * 1.1 / 4)
    camarilla_R3 = weekly_close + (H_minus_L * 1.1 / 4)
    
    # Align weekly pivots to 12h timeframe (shifted by 1 week to avoid look-ahead)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R3)
    
    # Get daily data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on daily close for trend filter
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema50_1d = daily_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = daily_close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align to 12h timeframe (shifted by 1 day to avoid look-ahead)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if required data not available
        if (np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from daily: up if EMA50 > EMA200, down if EMA50 < EMA200
        trend_up = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        trend_down = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price reaches R3 level (take profit)
            if close[i] >= camarilla_R3_aligned[i]:
                exit_long = True
            # Exit on trend reversal
            elif not trend_up:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price reaches S3 level (take profit)
            if close[i] <= camarilla_S3_aligned[i]:
                exit_short = True
            # Exit on trend reversal
            elif not trend_down:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S3 level, daily trend up, volume confirmation
            long_entry = (close[i] <= camarilla_S3_aligned[i] * 1.001) and trend_up and vol_confirm
            
            # Short entry: price touches R3 level, daily trend down, volume confirmation
            short_entry = (close[i] >= camarilla_R3_aligned[i] * 0.999) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals