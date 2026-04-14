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
    
    # Calculate 12h ATR (14-period) for volatility and stop
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # 1d Donchian channel (20-period) - breakout levels
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 1d volume average (20-period) for confirmation
    vol_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x 1d average
        vol_confirm = vol > (vol_1d_aligned[i] * 1.5) if not np.isnan(vol_1d_aligned[i]) else False
        
        if position == 0:
            # Long breakout: price breaks above 1d Donchian high + volume filter
            if price > donch_high_aligned[i] and vol_confirm and vol_filter:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below 1d Donchian low + volume filter
            elif price < donch_low_aligned[i] and vol_confirm and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below 1d Donchian low OR ATR-based stop
            if price < donch_low_aligned[i] or price < close[i-1] - 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above 1d Donchian high OR ATR-based stop
            if price > donch_high_aligned[i] or price > close[i-1] + 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dDonchian20_VolumeBreakout"
timeframe = "12h"
leverage = 1.0