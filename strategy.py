#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period high/low)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Align all indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current day's data (last completed day)
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed day
        
        if idx_1d < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current day's data
        high_20_today = high_20[idx_1d]
        low_20_today = low_20[idx_1d]
        ema50_today = ema50_1d[idx_1d]
        vol_today = df_1d['volume'].iloc[idx_1d]
        vol_avg_today = vol_avg_20[idx_1d]
        
        if np.isnan(high_20_today) or np.isnan(low_20_today) or np.isnan(ema50_today) or np.isnan(vol_today) or np.isnan(vol_avg_today):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 2.0x 20-period average
        vol_confirmed = vol_today > 2.0 * vol_avg_today
        
        # Current price
        price = close[i]
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                # Long when price breaks above Donchian high and above EMA50
                if price > high_20_today and price > ema50_today:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks below Donchian low and below EMA50
                elif price < low_20_today and price < ema50_today:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Manage long position
            exit_signal = False
            # Exit when price breaks below Donchian low
            if price < low_20_today:
                exit_signal = True
            # Exit when price crosses below EMA50
            elif price < ema50_today:
                exit_signal = True
            # Exit when volume confirmation lost
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Manage short position
            exit_signal = False
            # Exit when price breaks above Donchian high
            if price > high_20_today:
                exit_signal = True
            # Exit when price crosses above EMA50
            elif price > ema50_today:
                exit_signal = True
            # Exit when volume confirmation lost
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals