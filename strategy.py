#!/usr/bin/env python3
"""
Hypothesis: 1-hour EMA(20) pullback strategy with 4-hour trend filter and volume confirmation.
In uptrends (price > 4h EMA(50)), enter long on pullbacks to 1h EMA(20) with volume > 1.5x 20-bar average.
In downtrends (price < 4h EMA(50)), enter short on pullbacks to 1h EMA(20) with volume > 1.5x 20-bar average.
Uses 4h for trend direction, 1h only for entry timing. Session filter 08-20 UTC to avoid low-volume periods.
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1-hour EMA(20) for entry
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1-hour volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need 4h EMA, 1h EMA, and volume MA
    start_idx = max(50, 20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        trend_4h = ema_50_4h_aligned[i]
        ema_20_val = ema_20[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x 1h average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: EMA(20) pullback with volume and 4h trend alignment
        if position == 0:
            # Long: pullback to EMA(20) in 4h uptrend
            if close[i] > ema_20_val and close[i] > trend_4h and vol_filter:
                signals[i] = size
                position = 1
            # Short: pullback to EMA(20) in 4h downtrend
            elif close[i] < ema_20_val and close[i] < trend_4h and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below EMA(20) or trend reversal
            if close[i] < ema_20_val or close[i] < trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above EMA(20) or trend reversal
            if close[i] > ema_20_val or close[i] > trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_EMA20_Pullback_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0