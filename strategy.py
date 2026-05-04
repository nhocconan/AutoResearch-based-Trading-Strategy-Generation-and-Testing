#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# Uses 1d ATR(14) to filter low-volatility choppy regimes (ATR < 20-period SMA) and only trade in sufficient volatility
# Donchian(20) from prior 4h session provides clear breakout levels with structure
# Volume confirmation (>1.5x 20 EMA) ensures participation and reduces false breakouts
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 100-180 total trades over 4 years = 25-45/year for 4h.
# Works in both bull and bear: volatility filter avoids whipsaws in ranging markets, Donchian captures true breakouts.

name = "4h_Donchian20_1dATRVolFilter_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14) - exponential moving average of TR
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period SMA of ATR to define volatility regime
    atr_sma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR(14) > SMA(20) indicates sufficient volatility for breakout trading
    vol_filter = atr_14 > atr_sma_20
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter.astype(float))
    
    # Get 4h data for Donchian(20) calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) from prior 4h bar (using completed bar only)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower bands (20-period) - shift(1) for prior bar only
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe (completed 4h bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_filter_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + volatility filter + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                vol_filter_aligned[i] > 0.5 and  # Volatility filter active (boolean as float)
                volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + volatility filter + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_filter_aligned[i] > 0.5 and  # Volatility filter active
                  volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR volatility filter fails
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if close[i] < midpoint or vol_filter_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR volatility filter fails
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if close[i] > midpoint or vol_filter_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals