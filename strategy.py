#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1d ATR-based volatility regime filter with 12h Donchian channel breakout.
# Uses 1d ATR ratio (short/long) to identify low volatility regimes for breakout trading.
# Donchian breakout provides directional entry with volatility-based stops.
# Volume confirmation reduces false breakouts.
# Designed for low-frequency, high-quality trades to minimize fee drag in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR calculations
    atr_short = pd.Series(tr).ewm(span=7, adjust=False, min_periods=7).mean().values
    atr_long = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio (short/long) - identifies low volatility regimes
    atr_ratio = atr_short / atr_long
    
    # Load 12h data ONCE for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donch_period = 20
    upper_donch = pd.Series(high_12h).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low_12h).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align indicators to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    upper_donch_aligned = align_htf_to_ltf(prices, df_12h, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_12h, lower_donch)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20, 20)  # ATR long, Donchian, volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(upper_donch_aligned[i]) or
            np.isnan(lower_donch_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: low volatility (ATR ratio < 0.8)
        low_volatility = atr_ratio_aligned[i] < 0.8
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts in low volatility regimes
            # Long: price breaks above upper Donchian channel
            if (close[i] > upper_donch_aligned[i] and 
                low_volatility and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian channel
            elif (close[i] < lower_donch_aligned[i] and 
                  low_volatility and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower Donchian channel or volatility increases
            if (close[i] <= lower_donch_aligned[i] or 
                atr_ratio_aligned[i] > 1.2):  # High volatility
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to upper Donchian channel or volatility increases
            if (close[i] >= upper_donch_aligned[i] or 
                atr_ratio_aligned[i] > 1.2):  # High volatility
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dATRratio_12hDonchian_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0