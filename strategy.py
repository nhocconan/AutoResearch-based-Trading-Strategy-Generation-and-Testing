#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Channel_Breakout_Trend_1w"
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
    
    # Get weekly data for ATR calculation (longer period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR(10)
    high_low = df_1w['high'] - df_1w['low']
    high_close = np.abs(df_1w['high'] - df_1w['close'].shift())
    low_close = np.abs(df_1w['low'] - df_1w['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1w_4h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Get daily data for Keltner channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(20) for middle line
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Keltner Channels: EMA(20) ± 2 * ATR(1w)
    keltner_upper_1d = ema_20_1d + 2 * atr_1w
    keltner_lower_1d = ema_20_1d - 2 * atr_1w
    keltner_upper_4h = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_4h = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: above 1.8x 8-period average
    vol_ma = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 8  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_upper_4h[i]) or np.isnan(keltner_lower_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above upper Keltner with weekly uptrend
            if (close[i] > keltner_upper_4h[i] and 
                close[i] > ema_50_4h[i] and  # weekly uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below lower Keltner with weekly downtrend
            elif (close[i] < keltner_lower_4h[i] and 
                  close[i] < ema_50_4h[i] and  # weekly downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below EMA(20) (mean reversion to middle)
            if close[i] < ema_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above EMA(20) (mean reversion to middle)
            if close[i] > ema_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals