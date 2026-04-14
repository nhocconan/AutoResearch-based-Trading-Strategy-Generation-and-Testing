#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day ATR filter and volume confirmation
# Long when price breaks above 20-period Donchian upper channel AND 1-day ATR(14) > 20-period average ATR AND volume > 1.5x 20-period average volume
# Short when price breaks below 20-period Donchian lower channel AND 1-day ATR(14) > 20-period average ATR AND volume > 1.5x 20-period average volume
# Exit when price crosses back inside the Donchian channel (opposite band)
# Uses Donchian channels for breakout signals, ATR for volatility confirmation, volume for momentum confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 12h (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-day ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_avg = pd.Series(atr14_1d).rolling(window=20, min_periods=20).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr14_1d_avg_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d_avg)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(atr14_1d_avg_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: break above upper channel + ATR > average ATR + volume confirmation
            if (price > upper_channel[i] and atr14_1d_aligned[i] > atr14_1d_avg_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: break below lower channel + ATR > average ATR + volume confirmation
            elif (price < lower_channel[i] and atr14_1d_aligned[i] > atr14_1d_avg_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Donchian channel (below lower channel)
            if price < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside Donchian channel (above upper channel)
            if price > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_1dATR_Volume"
timeframe = "12h"
leverage = 1.0