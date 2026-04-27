#!/usr/bin/env python3
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
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily EMA(8) for entry filter
    ema8_1d = pd.Series(df_1d['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema8_1d_aligned = align_htf_to_ltf(prices, df_1d, ema8_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema8_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        ema_fast = ema8_1d_aligned[i]
        atr_current = atr[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_current > np.nanmean(atr[max(0, i-50):i+1]) * 0.8 if i >= 50 else True
        
        if position == 0:
            # Long: price above EMA8 and EMA34, with Donchian breakout
            if (close[i] > ema_fast and close[i] > ema_trend and 
                high[i] > high_20[i] and close[i] > high_20[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price below EMA8 and EMA34, with Donchian breakdown
            elif (close[i] < ema_fast and close[i] < ema_trend and 
                  low[i] < low_20[i] and close[i] < low_20[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below EMA8 or touches lower Donchian
            if close[i] < ema_fast or low[i] <= low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above EMA8 or touches upper Donchian
            if close[i] > ema_fast or high[i] >= high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA8_34_Donchian20_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0