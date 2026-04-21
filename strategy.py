#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly high/low from daily data (using 5-day rolling window for weekly)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low (5-day lookback)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    
    # Calculate weekly pivot points (using weekly high/low/close)
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # 60-period moving average for trend filter (6h timeframe)
    close = prices['close'].values
    ma_60 = pd.Series(close).rolling(window=60, min_periods=60).mean().values
    
    # Volume confirmation: 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ma_60[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        ma = ma_60[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol > 1.5 * vol_ma
        
        # Trend filter: price above/below 60-period MA
        uptrend = price > ma
        downtrend = price < ma
        
        if position == 0:
            # Long: Price crosses above weekly R1 + uptrend + volume confirmation
            if price > r1 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below weekly S1 + downtrend + volume confirmation
            elif price < s1 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Price crosses back below/above weekly pivot or trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below weekly pivot or trend reversal
                if price < pivot or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above weekly pivot or trend reversal
                if price > pivot or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0