#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Uses 1d ATR to filter breakouts by volatility regime (high volatility = better breakout follow-through)
# Entry logic: Long when price breaks above 6h Donchian upper band with volume spike and 1d ATR > 20-period median
#              Short when price breaks below 6h Donchian lower band with volume spike and 1d ATR > 20-period median
# Exit logic: Exit when price crosses the 6h Donchian middle band (mean reversion) or opposite band
# Works in both bull and bear markets by trading breakouts with volatility confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_Donchian20_Breakout_1dATR_Volume"
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
    
    # Calculate 1d ATR for volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(20) - Average True Range
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    # Median ATR for regime filter
    atr_median_1d = pd.Series(atr_20_1d).rolling(window=20, min_periods=20).median().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    
    # Calculate 6h Donchian channels (20-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian channels: upper = max(high,20), lower = min(low,20), middle = (upper+lower)/2
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to 6h timeframe (use previous completed 6h bar's levels)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr_20_1d_aligned[i]) or np.isnan(atr_median_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when current ATR > median ATR (high volatility regime)
        vol_filter = atr_20_1d_aligned[i] > atr_median_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 6h Donchian upper band AND volume spike AND volatility filter
            if (close[i] > donchian_upper_aligned[i] and 
                volume_spike[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 6h Donchian lower band AND volume spike AND volatility filter
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_spike[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 6h Donchian middle band (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 6h Donchian middle band (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals