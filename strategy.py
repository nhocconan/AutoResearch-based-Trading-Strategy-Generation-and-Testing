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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Close for price channel
    close_1d = df_1d['close'].values
    
    # Calculate 1d Highest High and Lowest Low for Donchian channel (20 period)
    highest_high_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    highest_high_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    lowest_low_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    
    # Calculate 1d ATR(14) for volatility filter and stop loss
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Volume moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high_1d_aligned[i]) or 
            np.isnan(lowest_low_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(volume_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = vol > (1.5 * volume_ma_1d_aligned[i])
        
        if position == 0:
            # Long setup: price breaks above 1d Donchian high + volume filter + volatility filter
            if (price > highest_high_1d_aligned[i] and volume_filter and vol_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below 1d Donchian low + volume filter + volatility filter
            elif (price < lowest_low_1d_aligned[i] and volume_filter and vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d Donchian low or ATR-based stop
            if (price < lowest_low_1d_aligned[i] or 
                price < (highest_high_1d_aligned[i] - 2.0 * atr[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d Donchian high or ATR-based stop
            if (price > highest_high_1d_aligned[i] or 
                price > (lowest_low_1d_aligned[i] + 2.0 * atr[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dDonchian20_Volume_ATRStop_v1"
timeframe = "12h"
leverage = 1.0