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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h Choppiness Index (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Avoid division by zero
    chop = np.where(range_max_min != 0, 
                    100 * np.log10(atr_sum / range_max_min) / np.log10(14), 
                    50)  # neutral when no range
    
    # Market regime: CHOP > 61.8 = range, CHOP < 38.2 = trend
    trending_market = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(trending_market[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volume spike + trending market
        breakout_long = close[i] > high_20[i]
        breakout_short = close[i] < low_20[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        trend_filter = trending_market[i]
        
        long_entry = breakout_long and vol_confirm and trend_filter
        short_entry = breakout_short and vol_confirm and trend_filter
        
        # Exit when price returns to opposite band (mean reversion)
        exit_long = position == 1 and close[i] < low_20[i]
        exit_short = position == -1 and close[i] > high_20[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_donchian_volume_chop_trend"
timeframe = "4h"
leverage = 1.0