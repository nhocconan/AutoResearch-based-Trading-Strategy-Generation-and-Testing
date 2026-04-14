#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(10) breakout + volume confirmation + 12h ATR volatility filter.
# Long when price breaks above 12h Donchian upper band AND 4h volume > 1.3x 20-period average AND 12h ATR(14) > 0.02 * close.
# Short when price breaks below 12h Donchian lower band AND 4h volume > 1.3x 20-period average AND 12h ATR(14) > 0.02 * close.
# Exit when price crosses back inside the 12h Donchian channel.
# This captures volatility-expansion breakouts with volume confirmation, filtering low-volatility false breakouts.
# Designed to work in both bull and bear markets by using volatility expansion as entry filter.
# Target: 20-50 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian channel and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 10-period Donchian channel on 12h
    donchian_up = np.full_like(high_12h, np.nan)
    donchian_down = np.full_like(low_12h, np.nan)
    for i in range(9, len(high_12h)):
        donchian_up[i] = np.max(high_12h[i-9:i+1])
        donchian_down[i] = np.min(low_12h[i-9:i+1])
    
    # Calculate 14-period ATR on 12h
    tr = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    atr_14 = np.full_like(close_12h, np.nan)
    for i in range(13, len(tr)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    atr_14 = np.insert(atr_14, 0, np.nan)  # align with close_12h index
    
    # Load 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume on 4h
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(19, len(volume_4h)):
        vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
    
    # Align indicators to 4h timeframe
    donchian_up_aligned = align_htf_to_ltf(prices, df_12h, donchian_up)
    donchian_down_aligned = align_htf_to_ltf(prices, df_12h, donchian_down)
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_up_aligned[i]) or 
            np.isnan(donchian_down_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        volume_ratio = volume_4h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        # Volatility filter: ATR > 2% of price
        vol_filter = atr_14_aligned[i] > 0.02 * close[i]
        
        if position == 0:
            # Look for entries: Donchian breakout + volume confirmation + volatility filter
            # Long: break above upper band AND volume > 1.3x average AND volatility filter
            if (close[i] > donchian_up_aligned[i] and 
                volume_ratio > 1.3 and 
                vol_filter):
                position = 1
                signals[i] = position_size
            # Short: break below lower band AND volume > 1.3x average AND volatility filter
            elif (close[i] < donchian_down_aligned[i] and 
                  volume_ratio > 1.3 and 
                  vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Donchian channel
            if close[i] < donchian_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside Donchian channel
            if close[i] > donchian_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_Volume_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0