#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly high/low from weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (using weekly high/low/close)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly pivot levels to 12h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Load daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average on 12h
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = vol > 1.8 * vol_ma
        
        # Trend filter: price above/below 50-day EMA
        uptrend = price > ema_50
        downtrend = price < ema_50
        
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

name = "12h_WeeklyPivot_R1_S1_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0