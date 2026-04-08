#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_fractal_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume context (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA for trend filter (50-period)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume SMA for volume context (20-period)
    vol_sma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR for volatility filter (14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d[i]) or np.isnan(volume_1d[i]) or 
            np.isnan(vol_sma_1d[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 4h bar
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)[i]
        vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)[i]
        
        # Trend filter: price above/below 50 EMA on 1d
        uptrend = close[i] > ema_1d_aligned
        downtrend = close[i] < ema_1d_aligned
        
        # Volume filter: current volume above 1.5x 1d average volume
        volume_filter = volume[i] > (vol_sma_1d_aligned * 1.5)
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band OR trend reversal
            if close[i] < lowest_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band OR trend reversal
            if close[i] > highest_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper band + uptrend + volume filter
            if close[i] > highest_high[i] and uptrend and volume_filter:
                position = 1
                signals[i] = 0.30
            # Short: price breaks below Donchian lower band + downtrend + volume filter
            elif close[i] < lowest_low[i] and downtrend and volume_filter:
                position = -1
                signals[i] = -0.30
    
    return signals