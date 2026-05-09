#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Triangle_Squeeze_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Daily ATR for volatility contraction detection
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_shift = np.roll(daily_close, 1)
    daily_close_shift[0] = daily_close[0]
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - daily_close_shift)
    tr3 = np.abs(daily_low - daily_close_shift)
    daily_tr = np.maximum(tr1, np.maximum(tr2, tr3))
    daily_atr = pd.Series(daily_tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    daily_atr_6h = align_htf_to_ltf(prices, df_1d, daily_atr)
    
    # 60-period high/low for triangle breakout (2.5 days on 6h)
    high_max = pd.Series(high).rolling(window=60, min_periods=60).max().values
    low_min = pd.Series(low).rolling(window=60, min_periods=60).min().values
    
    # Volume filter: above 1.8x 40-period average
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(daily_ema_6h[i]) or np.isnan(daily_atr_6h[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma[i]
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        # Triangle squeeze condition: volatility contraction
        # Current ATR < 0.7 * 20-period ATR average
        if i >= 20:
            atr_ma = np.nanmean(daily_atr_6h[i-20:i])
            squeeze_ok = daily_atr_6h[i] < 0.7 * atr_ma
        else:
            squeeze_ok = False
        
        if position == 0:
            # Long breakout: price breaks above 60-period high with uptrend and squeeze
            if (close[i] > high_max[i] and 
                close[i] > daily_ema_6h[i] and  # daily uptrend
                vol_ok and 
                squeeze_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 60-period low with downtrend and squeeze
            elif (close[i] < low_min[i] and 
                  close[i] < daily_ema_6h[i] and  # daily downtrend
                  vol_ok and 
                  squeeze_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 60-period low or trend reverses
            if (close[i] < low_min[i] or 
                close[i] < daily_ema_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 60-period high or trend reverses
            if (close[i] > high_max[i] or 
                close[i] > daily_ema_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals