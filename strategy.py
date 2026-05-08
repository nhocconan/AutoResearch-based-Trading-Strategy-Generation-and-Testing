#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_Trend_Volume_1d"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d TRIX (15-period)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = (ema3 - previous ema3) / previous ema3 * 100
    trix_raw = np.full(len(ema3), np.nan)
    for i in range(1, len(ema3)):
        if ema3[i-1] != 0:
            trix_raw[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Align 1d data to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for TRIX
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(trix_aligned[i]) or np.isnan(vol_avg_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 1d bar's data (last completed 1d bar)
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed 1d bar
        
        if idx_1d < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_current = trix_raw[idx_1d]
        vol_avg_20_current = vol_avg_20[idx_1d]
        
        if np.isnan(trix_current) or np.isnan(vol_avg_20_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_current = df_1d['volume'].iloc[idx_1d]
        vol_confirmed = vol_current > 1.5 * vol_avg_20_current
        
        # TRIX signal: positive for long, negative for short
        trix_positive = trix_current > 0
        trix_negative = trix_current < 0
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                # Long when TRIX positive
                if trix_positive:
                    signals[i] = 0.25
                    position = 1
                # Short when TRIX negative
                elif trix_negative:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Manage long position
            exit_signal = False
            # Exit when TRIX turns negative or volume confirmation lost
            if not trix_positive:
                exit_signal = True
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
            # Exit when TRIX turns positive or volume confirmation lost
            if not trix_negative:
                exit_signal = True
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals