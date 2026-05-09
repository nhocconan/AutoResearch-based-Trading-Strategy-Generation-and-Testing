#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_EMA_Ratio_Trend_With_Volume"
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
    
    # Get daily data for EMA ratio trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA ratio: EMA(8)/EMA(21) on daily close
    ema_8_1d = pd.Series(df_1d['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_ratio_1d = ema_8_1d / ema_21_1d
    
    # Align EMA ratio to 6h
    ema_ratio_6h = align_htf_to_ltf(prices, df_1d, ema_ratio_1d)
    
    # Volume filter: above 1.5x 24-period average (24*6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA and EMA ratio
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_ratio_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: EMA ratio > 1.0 (uptrend) with volume confirmation
            if (ema_ratio_6h[i] > 1.0 and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: EMA ratio < 0.98 (downtrend) with volume confirmation
            elif (ema_ratio_6h[i] < 0.98 and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: EMA ratio falls back below 1.0 (trend weakening)
            if ema_ratio_6h[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA ratio rises back above 0.98 (trend weakening)
            if ema_ratio_6h[i] > 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals