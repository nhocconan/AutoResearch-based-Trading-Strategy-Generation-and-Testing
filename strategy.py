#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Refined"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for each day
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    R3 = close_d + (high_d - low_d) * 1.1 / 2
    S3 = close_d - (high_d - low_d) * 1.1 / 2
    
    # Calculate daily volume average (30-period)
    vol_1d = df_1d['volume'].values
    vol_avg_30 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 29:
            vol_avg_30[i] = np.mean(vol_1d[i-29:i+1])
    
    # Align all indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    vol_avg_30_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_avg_30_aligned[i])):
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
        ema34_today = ema34_1d[idx_1d]
        R3_today = R3[idx_1d]
        S3_today = S3[idx_1d]
        vol_today = df_1d['volume'].iloc[idx_1d]
        vol_avg_today = vol_avg_30[idx_1d]
        
        if np.isnan(ema34_today) or np.isnan(R3_today) or np.isnan(S3_today) or np.isnan(vol_today) or np.isnan(vol_avg_today):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 30-period average
        vol_confirmed = vol_today > 1.5 * vol_avg_today
        
        # Current price
        price = close[i]
        
        # Trading logic
        if position == 0:
            # Look for entry - only trade in direction of daily trend
            if vol_confirmed:
                # Long when price breaks above R3 and above EMA34 (bullish trend)
                if price > R3_today and price > ema34_today:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks below S3 and below EMA34 (bearish trend)
                elif price < S3_today and price < ema34_today:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Manage long position
            exit_signal = False
            # Exit when price breaks below S3
            if price < S3_today:
                exit_signal = True
            # Exit when price crosses below EMA34
            elif price < ema34_today:
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
            # Exit when price breaks above R3
            if price > R3_today:
                exit_signal = True
            # Exit when price crosses above EMA34
            elif price > ema34_today:
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