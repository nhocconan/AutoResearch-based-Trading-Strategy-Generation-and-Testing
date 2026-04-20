#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h chart with 1d Donchian breakout and 1w KAMA trend filter.
# Long when price breaks above 20-period 1d Donchian high with price above 1w KAMA.
# Short when price breaks below 20-period 1d Donchian low with price below 1w KAMA.
# Uses weekly KAMA to filter trend direction and avoid counter-trend trades.
# Target: 20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period Donchian channels (previous day's to avoid look-ahead)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous day's values
    donchian_high = np.roll(high_20, 1)
    donchian_low = np.roll(low_20, 1)
    donchian_high[0] = high_1d[0]
    donchian_low[0] = low_1d[0]
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load 1w data for KAMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = pd.Series(er).rolling(window=er_len, min_periods=1).mean().values
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(kama_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_donchian = donchian_high_aligned[i]
        lower_donchian = donchian_low_aligned[i]
        kama_val = kama_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, above KAMA, volume
            if price > upper_donchian and price > kama_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below KAMA, volume
            elif price < lower_donchian and price < kama_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or below KAMA
            if price < lower_donchian or price < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or above KAMA
            if price > upper_donchian or price > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian20_1w_KAMA_TrendFilter_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0