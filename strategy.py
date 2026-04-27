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
    
    # Get 1d data for daily ATR and volume (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily volume average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        atr_val = atr_14_1d_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_current = volume[i]
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        if i >= 20:
            atr_ma = pd.Series(atr_14_1d_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1]
        else:
            atr_ma = atr_val
        vol_filter = atr_val > atr_ma
        
        # Volume filter: current volume > 1.5x daily average
        volume_filter = vol_current > (vol_avg * 1.5)
        
        # Calculate daily high/low for price action
        # Map 6h index to daily index (4 6h bars per day)
        daily_idx = i // 4
        if daily_idx >= len(high_1d):
            signals[i] = 0.0
            continue
            
        daily_high = high_1d[daily_idx]
        daily_low = low_1d[daily_idx]
        daily_range = daily_high - daily_low
        
        if position == 0:
            # Long: price breaks above daily high with volume and volatility
            if close[i] > daily_high and vol_filter and volume_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below daily low with volume and volatility
            elif close[i] < daily_low and vol_filter and volume_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below daily close or volatility drops
            daily_close = close_1d[daily_idx]
            if close[i] < daily_close or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above daily close or volatility drops
            daily_close = close_1d[daily_idx]
            if close[i] > daily_close or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_DailyBreakout_VolumeVolatilityFilter"
timeframe = "6h"
leverage = 1.0