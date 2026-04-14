#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy with 1d Donchian breakout + volume confirmation + KAMA trend filter
# Donchian breakout captures strong directional moves in both bull and bear markets
# Volume confirmation ensures breakouts are genuine and not false signals
# KAMA adapts to market conditions, reducing whipsaw in choppy markets
# Uses 1d Donchian channels for breakout direction, 1d volume spike for confirmation,
# and 12h KAMA for trend filtering - aims for low trade frequency with high edge

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (20 periods)
    donch_len = 20
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper and lower bands
    upper = pd.Series(high_1d).rolling(window=donch_len, min_periods=donch_len).max().values
    lower = pd.Series(low_1d).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Align Donchian to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Calculate 1d volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Calculate 12h KAMA for trend filter
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Seed value
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, donch_len + 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(kama[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band + volume spike + price > KAMA
            if price > upper_aligned[i] and volume_spike and price > kama[i]:
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian band + volume spike + price < KAMA
            elif price < lower_aligned[i] and volume_spike and price < kama[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian band OR volume drops significantly
            if price < lower_aligned[i] or vol < 0.5 * vol_ma_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above upper Donchian band OR volume drops significantly
            if price > upper_aligned[i] or vol < 0.5 * vol_ma_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dDonchian_Volume_KAMA_v1"
timeframe = "12h"
leverage = 1.0