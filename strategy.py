#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with 1d volume spike filter and 1d trend filter (EMA34)
# Long when TRIX crosses above zero + volume > 1.5x 20-day average + close > daily EMA34
# Short when TRIX crosses below zero + volume > 1.5x 20-day average + close < daily EMA34
# Exit when TRIX crosses back through zero
# TRIX (triple exponential smoothing) filters noise and catches momentum shifts
# Volume spike confirms institutional interest; EMA34 filter ensures trend alignment
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_TRIX_Momentum_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-day average volume for volume spike filter
    vol_ma_20 = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate TRIX on 4h close: triple EMA of close, then % change
    # EMA1 = EMA(close, 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 = EMA(EMA1, 12)
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 = EMA(EMA2, 12)
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # First value remains 0 (no previous)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or np.isnan(trix[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day average
        # Find the most recent completed daily bar
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed daily bar
        
        if idx_1d < 0:
            vol_filter = False
        else:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            vol_filter = vol_1d_current > 1.5 * vol_ma_20_aligned[i] if not pd.isnan(vol_ma_20_aligned[i]) else False
        
        if position == 0:
            # Look for entry: TRIX crossover + volume + trend
            long_condition = trix[i] > 0 and trix[i-1] <= 0 and vol_filter and close[i] > ema_34_aligned[i]
            short_condition = trix[i] < 0 and trix[i-1] >= 0 and vol_filter and close[i] < ema_34_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses back below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses back above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals