#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RVI_Signal_1dTrend_ETF_Style"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and ETF-style RVI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RVI (Relative Vigor Index) on 6h timeframe
    # RVI = (Close - Open) / (High - Low) smoothed
    num = close - prices['open'].values
    den = high - low
    # Avoid division by zero
    den = np.where(den == 0, 1e-10, den)
    rvi_raw = num / den
    
    # Smooth RVI with 10-period SMA
    rvi_series = pd.Series(rvi_raw)
    rvi = rvi_series.rolling(window=10, min_periods=10).mean().values
    
    # Signal line: EMA of RVI
    rvi_ema = pd.Series(rvi).ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: above average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rvi[i]) or np.isnan(rvi_ema[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: RVI crosses above signal line + above 1d EMA + volume
            if (rvi[i] > rvi_ema[i] and 
                rvi[i-1] <= rvi_ema[i-1] and  # crossover
                close[i] > ema_1d_aligned[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: RVI crosses below signal line + below 1d EMA + volume
            elif (rvi[i] < rvi_ema[i] and 
                  rvi[i-1] >= rvi_ema[i-1] and  # crossover
                  close[i] < ema_1d_aligned[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RVI crosses below signal line
            if rvi[i] < rvi_ema[i] and rvi[i-1] >= rvi_ema[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RVI crosses above signal line
            if rvi[i] > rvi_ema[i] and rvi[i-1] <= rvi_ema[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals