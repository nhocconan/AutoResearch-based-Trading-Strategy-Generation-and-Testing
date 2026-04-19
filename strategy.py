#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d VWAP trend and volume confirmation.
# Uses 1d VWAP for trend direction and 12h Donchian breakout for momentum.
# Enters only during 08-20 UTC session to avoid low-volume noise.
# Targets 15-37 trades/year (60-150 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trends.
name = "12h_1d_VWAP_Donchian20_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for VWAP trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    # Calculate VWAP: cumulative(volume * price) / cumulative(volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / vwap_den
    vwap_1d = np.where(vwap_den == 0, np.nan, vwap_1d)  # Avoid division by zero
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Get 12h data for Donchian20 breakout (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    # Donchian channels: 20-period high/low
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(high_20_12h_aligned[i]) or 
            np.isnan(low_20_12h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d VWAP AND breaks 12h Donchian high with volume
            if (close[i] > vwap_1d_aligned[i] and 
                close[i] > high_20_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below 1d VWAP AND breaks 12h Donchian low with volume
            elif (close[i] < vwap_1d_aligned[i] and 
                  close[i] < low_20_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d VWAP or 12h Donchian low
            if close[i] < vwap_1d_aligned[i] or close[i] < low_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above 1d VWAP or 12h Donchian high
            if close[i] > vwap_1d_aligned[i] or close[i] > high_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals