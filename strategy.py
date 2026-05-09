#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_RVI_Signal_4hTrend_1dFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RVI on 1h timeframe
    num = close - open_price
    den = high - low
    den = np.where(den == 0, 1e-10, den)
    rvi_raw = num / den
    
    # Smooth RVI with 10-period SMA
    rvi_series = pd.Series(rvi_raw)
    rvi = rvi_series.rolling(window=10, min_periods=10).mean().values
    
    # Signal line: EMA of RVI
    rvi_ema = pd.Series(rvi).ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # 4h EMA50 for trend filter
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA200 for long-term filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rvi[i]) or np.isnan(rvi_ema[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        in_session = (8 <= hours[i] <= 20)
        
        if position == 0:
            # Long: RVI crosses above signal line + above 4h EMA + above 1d EMA200 + volume + session
            if (rvi[i] > rvi_ema[i] and 
                rvi[i-1] <= rvi_ema[i-1] and  # crossover
                close[i] > ema_4h_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: RVI crosses below signal line + below 4h EMA + below 1d EMA200 + volume + session
            elif (rvi[i] < rvi_ema[i] and 
                  rvi[i-1] >= rvi_ema[i-1] and  # crossover
                  close[i] < ema_4h_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RVI crosses below signal line or price below 4h EMA
            if (rvi[i] < rvi_ema[i] and rvi[i-1] >= rvi_ema[i-1]) or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RVI crosses above signal line or price above 4h EMA
            if (rvi[i] > rvi_ema[i] and rvi[i-1] <= rvi_ema[i-1]) or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals