#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H4, L4) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h4 = close_1d + range_1d * 1.1 / 2
    l4 = close_1d - range_1d * 1.1 / 2
    
    # Align H4 and L4 to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 1d volatility (ATR-like: average true range over 10 days)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure volatility and volume averages are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Session filter: 08-20 UTC (more active hours)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        # Volatility filter: current 1d volatility > 1.3 * median of last 30 periods
        vol_filter = atr_aligned[i] > 1.3 * np.nanmedian(atr_aligned[max(0, i-30):i])
        
        # Volume filter: current volume > 1.8 * 1d average volume (higher threshold for fewer trades)
        vol_surge = volume[i] > 1.8 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks through Camarilla H4/L4 with volatility and volume surge
        long_entry = (high[i] > h4_aligned[i] and vol_filter and vol_surge and in_session)
        short_entry = (low[i] < l4_aligned[i] and vol_filter and vol_surge and in_session)
        
        # Exit conditions: price returns to pivot level (more conservative exit)
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        exit_low = low[i] < pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else False
        exit_high = high[i] > pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_low:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_high:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals